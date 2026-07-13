"""Realización PINN del operador de continuación lateral

    Λ : (g, f) ↦ T,      ρc ∂ₜT = ∇·(k ∇T)   en  Q = Ω × (0, Tmax],
                         Ω = D × (0, L),  D = disco de radio R,

con dato de Cauchy sobre la tapa Γ_sup = D × {L} y flujo lateral nulo:

    T|_Γsup = g,   −k ∂_z T|_Γsup = f,   ∂_n T|_Γlat = 0;
    Γ₀ = D × {0} libre;  sin condición inicial en t = 0.

Tras `fit`, la instancia expone las vistas del operador entrenado

    T      : X ↦ T(X)            (N,4) → (N,1)
    grad_T : X ↦ ∇T(X)           (N,4) → (N,3)     ← objetivo práctico
    flux   : X ↦ −k(X) ∇T(X)     (N,4) → (N,3)

Planteamiento riguroso y unicidad (Proposición 1): artículo asociado (no
distribuido en este repositorio); resumen operativo autocontenido en CLAUDE.md §1.
"""

import math
import warnings

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)  # doble precision: importa en PINNs


class _Net(nn.Module):
    """MLP tanh con entrada normalizada  H = 2(X − lb)/(ub − lb) − 1 ∈ [−1,1]⁴."""

    def __init__(self, layers, lb, ub):
        super().__init__()
        mods = []
        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            mods.append(lin)
            if i < len(layers) - 2:
                mods.append(nn.Tanh())  # derivadas suaves; NO ReLU
        self.seq = nn.Sequential(*mods)
        self.register_buffer("lb", lb)
        self.register_buffer("ub", ub)

    def forward(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.seq(H)


def _grad(u, x):
    """Derivada de u respecto a x manteniendo el grafo (para orden superior)."""
    return torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]


def _not_fitted(*_a, **_k):
    raise RuntimeError("Modelo no entrenado: llama a fit(g, f) antes de usar grad_T/T/flux.")


class LateralCauchyCylinder:
    """Λ : (g, f) ↦ T para un medio (ρ, c, k) fijo.

    La instancia ES el operador: la geometría (R, L, Tmax) y el medio van en el
    constructor porque lo definen; el dato (g, f) entra en `fit` porque es su
    argumento. Tras `fit`: T = Λ(g,f), grad_T = ∇T, flux = −k∇T, α = k/(ρc)."""

    def __init__(self, R, L, Tmax, rho, c, k, net_config=None):
        self.R, self.L, self.Tmax = float(R), float(L), float(Tmax)
        self.rho, self.c, self.k = rho, c, k
        self.rhoc = lambda X: rho(X) * c(X)
        self.alpha = lambda X: k(X) / (rho(X) * c(X))  # difusividad (diagnostico)

        cfg = {
            "layers": [4, 96, 96, 96, 96, 1],
            "activation": "tanh",
            "seed": 0,
            "fourier_features": False,  # apagadas: amplifican lo mal puesto
        }
        if net_config:
            cfg.update(net_config)
        if cfg["fourier_features"]:
            raise NotImplementedError("fourier_features no implementado (ver §3).")
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lb = torch.tensor([-self.R, -self.R, 0.0, 0.0], device=self.device)
        self.ub = torch.tensor([self.R, self.R, self.L, self.Tmax], device=self.device)
        self._build_net()

        # salidas: aun no entrenado (error claro si se invocan antes de fit)
        self.grad_T = self.T = self.flux = _not_fitted

    # ------------------------------------------------------------------ red
    def _build_net(self):
        torch.manual_seed(self.cfg["seed"])
        self.net = _Net(self.cfg["layers"], self.lb, self.ub).to(self.device)

    # -------------------------------------------------------------- muestreo
    def _sample(self, n_int, n_top, n_lat):
        """Colocacion en el disco de radio R (aislado para un D general futuro)."""
        R, L, Tmax = self.R, self.L, self.Tmax

        def disk(n, z=None):
            r = R * torch.sqrt(torch.rand(n, 1))      # sqrt -> uniforme en area
            th = 2 * math.pi * torch.rand(n, 1)
            x, y = r * torch.cos(th), r * torch.sin(th)
            zz = L * torch.rand(n, 1) if z is None else torch.full((n, 1), z)
            t = Tmax * torch.rand(n, 1)
            return torch.cat([x, y, zz, t], 1).to(self.device)

        Xi = disk(n_int)            # interior
        Xt = disk(n_top, z=L)       # tapa z=L
        th = 2 * math.pi * torch.rand(n_lat, 1)       # pared lateral r=R
        Xl = torch.cat([
            R * torch.cos(th), R * torch.sin(th),
            L * torch.rand(n_lat, 1), Tmax * torch.rand(n_lat, 1),
        ], 1).to(self.device)
        return Xi, Xt, Xl

    # ------------------------------------------------------------- residuo
    def _pde_residual(self, X):
        """r(X) = ρc ∂ₜT − ∇·(k ∇T), en forma divergencia (doble autodiff)."""
        X = X.clone().requires_grad_(True)
        T = self.net(X)
        TX = _grad(T, X)                      # [T_x, T_y, T_z, T_t]
        flux = self.k(X) * TX[:, 0:3]         # k grad T (espacial)
        div = sum(_grad(flux[:, i:i + 1], X)[:, i:i + 1] for i in range(3))
        return self.rhoc(X) * TX[:, 3:4] - div

    # ----------------------------------------------------------------- fit
    def fit(self, g, f, **opts):
        """Entrena la red minimizando

            L(θ) = λ_pde·‖r‖² + λ_g·‖T−g‖²_Γsup + λ_f·‖−k∂_zT−f‖²_Γsup
                   + λ_lat·‖∂_nT‖²_Γlat        (medias empíricas; Γ₀ SIN término)

        con g, f BLANDOS (nunca Dirichlet duro). Adam → L-BFGS. Devuelve el
        history de pérdidas por término e instala T, grad_T y flux."""
        o = {
            "weights": (1.0, 10.0, 10.0, 1.0),  # (pde, g, f, lat)
            "adam_iters": 15000, "lbfgs_iters": 3000, "lr": 1e-3,
            "n_int": 8000, "n_top": 3000, "n_lat": 2000,
            "resample_every": 500,   # resamplea colocacion cada K iters de Adam (0=off)
        }
        o.update(opts)
        self._build_net()  # re-fit reinicia (sin warm-start)
        lp, lg, lf, ll = o["weights"]
        Xi, Xt, Xl = self._sample(o["n_int"], o["n_top"], o["n_lat"])
        hist = {"total": [], "pde": [], "g": [], "f": [], "lat": []}

        def terms():
            Lp = (self._pde_residual(Xi) ** 2).mean()
            Xt_ = Xt.clone().requires_grad_(True)
            Tt = self.net(Xt_)
            TXt = _grad(Tt, Xt_)
            Lg = ((Tt - g(Xt)) ** 2).mean()                       # Dirichlet blando
            Lf = ((-self.k(Xt) * TXt[:, 2:3] - f(Xt)) ** 2).mean()  # 2do dato Cauchy
            Xl_ = Xl.clone().requires_grad_(True)
            TXl = _grad(self.net(Xl_), Xl_)
            dn = (Xl[:, 0:1] / self.R) * TXl[:, 0:1] + (Xl[:, 1:2] / self.R) * TXl[:, 1:2]
            Ll = (dn ** 2).mean()                                 # Neumann lateral
            return Lp, Lg, Lf, Ll

        def record(loss, Lp, Lg, Lf, Ll):
            for key, v in zip(("total", "pde", "g", "f", "lat"),
                              (loss, Lp, Lg, Lf, Ll)):
                hist[key].append(float(v))

        # --- Adam: paso explicito (no usa closure); UN registro por iteracion ---
        adam = torch.optim.Adam(self.net.parameters(), lr=o["lr"])
        for it in range(o["adam_iters"]):
            adam.zero_grad()
            Lp, Lg, Lf, Ll = terms()
            loss = lp * Lp + lg * Lg + lf * Lf + ll * Ll
            loss.backward()
            adam.step()
            record(loss.item(), Lp.item(), Lg.item(), Lf.item(), Ll.item())
            # resamplear refresca el residuo (NUNCA durante L-BFGS: rompe su Hessiano)
            if o["resample_every"] and (it + 1) % o["resample_every"] == 0:
                Xi, Xt, Xl = self._sample(o["n_int"], o["n_top"], o["n_lat"])

        # --- L-BFGS: colocacion FIJA; closure solo aqui; UN registro por iteracion ---
        last = {}

        def closure():
            lbfgs.zero_grad()
            Lp, Lg, Lf, Ll = terms()
            loss = lp * Lp + lg * Lg + lf * Lf + ll * Ll
            loss.backward()
            last["vals"] = (loss.item(), Lp.item(), Lg.item(), Lf.item(), Ll.item())
            return loss

        lbfgs = torch.optim.LBFGS(
            self.net.parameters(), max_iter=1,   # 1 iter por step -> registro limpio
            line_search_fn="strong_wolfe", tolerance_grad=1e-9,
            tolerance_change=1e-12, history_size=50,
        )
        prev = float("inf")
        for _ in range(o["lbfgs_iters"]):
            lbfgs.step(closure)
            record(*last["vals"])
            cur = last["vals"][0]
            if abs(prev - cur) < 1e-12:          # convergio: corta (lo hacia max_iter)
                break
            prev = cur

        self._install()
        self.history = hist                      # accesible luego via op.history
        return hist

    # ----------------------------------------------- validación y reporte
    def validate(self, truth, n=3000, seed=0, nbins=6):
        """Valida ∇T contra una verdad `truth` (objeto con grad_T: exacta
        manufacturada o ReferenceSolution). Devuelve dict con

            err_base   = e en la base z=0   ← métrica principal
            err_global = e en todo Ω        (dominado por la zona fácil)
            z = (zc, e(z)),  t = (tc, e(t))  perfiles para plotting.*

        donde e = ‖∇T_θ − ∇T‖/‖∇T‖."""
        from .metrics import (sample_cylinder, sample_disk_slice, rel_l2,
                              error_vs_z, error_vs_t)
        X = sample_cylinder(self.R, self.L, self.Tmax, n, seed=seed)
        Xb = sample_disk_slice(self.R, self.Tmax, 0.0, n, seed=seed + 1)
        P = self.grad_T(torch.as_tensor(X, device=self.device))
        Pb = self.grad_T(torch.as_tensor(Xb, device=self.device))
        Gt, Gtb = truth.grad_T(X), truth.grad_T(Xb)
        return {
            "err_base": rel_l2(Pb, Gtb),
            "err_global": rel_l2(P, Gt),
            "z": error_vs_z(P, Gt, X, nbins),
            "t": error_vs_t(P, Gt, X, nbins),
        }

    def plot_history(self, path=None, title=None):
        """Figura de la última corrida de fit (estilo artículo; ver plotting)."""
        if not hasattr(self, "history"):
            raise RuntimeError("Sin history: llama a fit antes de plot_history.")
        from .plotting import plot_history
        return plot_history(self.history, path=path, title=title)

    # --------------------------------------------------- salidas invocables
    def _prep(self, X, grad=False):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        X = X.to(self.device, torch.get_default_dtype())
        self._domain_check(X)
        return X.clone().requires_grad_(True) if grad else X

    def _domain_check(self, X):
        r = torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        out = ((r > self.R + 1e-6) | (X[:, 2] < -1e-6) | (X[:, 2] > self.L + 1e-6)
               | (X[:, 3] < -1e-6) | (X[:, 3] > self.Tmax + 1e-6))
        if out.any():
            warnings.warn(f"{int(out.sum())} puntos fuera de Omega x (0, Tmax].")

    def _install(self):
        def T(X, create_graph=False):
            out = self.net(self._prep(X))
            return out if create_graph else out.detach()

        def grad_T(X, create_graph=False):
            Xg = self._prep(X, grad=True)
            out = self.net(Xg)
            gr = torch.autograd.grad(out, Xg, torch.ones_like(out),
                                     create_graph=create_graph)[0][:, 0:3]
            return gr if create_graph else gr.detach()

        def flux(X, create_graph=False):
            Xg = self._prep(X, grad=True)
            out = self.net(Xg)
            gr = torch.autograd.grad(out, Xg, torch.ones_like(out),
                                     create_graph=create_graph)[0][:, 0:3]
            fl = -self.k(Xg) * gr
            return fl if create_graph else fl.detach()

        self.T, self.grad_T, self.flux = T, grad_T, flux

    # ---------------------------------------------------- guardar / cargar
    def save(self, path):
        """Guarda pesos + geometria + net_config. El medio (rho, c, k) NO se
        serializa (son callables): hay que volver a pasarlo en load."""
        torch.save({
            "state_dict": self.net.state_dict(),
            "geometry": (self.R, self.L, self.Tmax),
            "net_config": self.cfg,
        }, path)

    @classmethod
    def load(cls, path, rho, c, k, map_location=None):
        """Reconstruye un operador entrenado. Requiere el MISMO medio (rho, c, k)
        que lo definio, porque define el operador y no se puede serializar."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        R, L, Tmax = ckpt["geometry"]
        op = cls(R, L, Tmax, rho, c, k, net_config=ckpt["net_config"])
        op.net.load_state_dict(ckpt["state_dict"])
        op._install()      # reinstala grad_T / T / flux (modelo ya entrenado)
        return op
