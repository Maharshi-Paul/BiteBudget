"""
EATING OUTSIDE  vs  MONTHLY FOOD EXPENSE
Simulation / project for  "Mathematical Foundations for AI & Data Science"  (JIIT)
===============================================================================
Run:   python eating_out_vs_food_expense.py
Needs: numpy, pandas, matplotlib, tkinter   (no scipy needed - everything is coded by hand)

SYLLABUS MAP (one tab per block of lectures)
  Tab 1  Data & Simulator ........ survey generator (Poisson counts + noise) / load your real Google-Form data
  Tab 2  Probability & Bayes ..... Lec 2-4  : probability, conditional probability, total probability, Bayes
  Tab 3  Distributions & MGF ..... Lec 5-13 : random variables, Binomial, Poisson, moments, MGF
  Tab 4  Normal & CLT ............ Lec 12   : Normal distribution, z-scores, Central Limit Theorem (bootstrap sim)
  Tab 5  Correlation ............. Pearson, Spearman rank correlation, permutation test
  Tab 6  Regression + Lin. Alg. .. Lec 16-22: least squares, vector space, subspace, basis, dimension, projection,
                                              linear transformation (rank-nullity)
  Tab 7  LPP (current chapters) .. Lec 31-35: LPP formulation, Simplex, Duality, Convex sets, sensitivity
"""
import math
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

MEALS_PER_MONTH = 90          # 30 days x 3 meals


# =====================================================================================
#  PART A : PURE MATHS (no GUI)  - every formula from the syllabus is coded here
# =====================================================================================
def generate_survey(n=200, seed=42, base=4500.0, inc_eff=0.03, cost=100.0, noise=500.0):
    """Synthetic survey.  eat_out ~ Poisson(lambda_i) with lambda_i depending on income & hostel,
    expense = base + inc_eff*income + cost*eat_out + Normal(0, noise)."""
    rng = np.random.default_rng(seed)
    hostel = rng.binomial(1, 0.55, n)
    income = np.clip(rng.normal(22000, 7000, n), 5000, None)
    lam = np.exp(2.4 + 0.00004 * (income - 22000) + 0.35 * hostel)
    eat = np.minimum(rng.poisson(lam), MEALS_PER_MONTH)
    exp_ = base + inc_eff * income + cost * eat + rng.normal(0, noise, n)
    return pd.DataFrame({"id": np.arange(1, n + 1), "income": income.round(0), "hostel": hostel,
                         "eat_out": eat.astype(int), "expense": exp_.round(0)})


def prob_table(df, k, t):
    """F = frequent eater (eat_out >= k), H = high expense (expense >= t)."""
    F = (df.eat_out >= k).values
    H = (df.expense >= t).values
    n = len(df)
    nF, nH, nFH, nFcH = F.sum(), H.sum(), (F & H).sum(), ((~F) & H).sum()
    pF, pH = nF / n, nH / n
    pH_F = nFH / nF if nF else np.nan
    pH_Fc = nFcH / (n - nF) if n - nF else np.nan
    pH_total = pH_F * pF + pH_Fc * (1 - pF)                      # law of total probability
    pF_H_bayes = pH_F * pF / pH_total if pH_total else np.nan     # Bayes theorem
    pF_H_direct = nFH / nH if nH else np.nan
    return dict(n=n, nF=nF, nH=nH, nFH=nFH, nFcH=nFcH, pF=pF, pH=pH, pH_F=pH_F, pH_Fc=pH_Fc,
                pH_total=pH_total, pF_H_bayes=pF_H_bayes, pF_H_direct=pF_H_direct,
                pFH=nFH / n, F=F, H=H)


def poisson_pmf(k, lam):
    k = np.asarray(k)
    return np.exp(k * math.log(lam) - lam - np.array([math.lgamma(x + 1) for x in k]))


def binom_pmf(k, n, p):
    k = np.asarray(k)
    lc = np.array([math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1) for x in k])
    return np.exp(lc + k * math.log(p) + (n - k) * math.log(1 - p))


def normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def normal_cdf(x, mu, sd):
    return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))


def moments(x):
    x = np.asarray(x, float)
    m = x.mean()
    raw = [np.mean(x ** k) for k in range(1, 5)]
    cen = [np.mean((x - m) ** k) for k in range(1, 5)]
    var = cen[1]
    return dict(raw=raw, cen=cen, mean=m, var=var, skew=cen[2] / var ** 1.5, kurt=cen[3] / var ** 2 - 3)


def ols_simple(x, y):
    """slope/intercept by least squares (Lec 16)."""
    sxx = np.sum((x - x.mean()) ** 2)
    b1 = np.sum((x - x.mean()) * (y - y.mean())) / sxx
    return b1, y.mean() - b1 * x.mean()


def estimate_prices(df):
    """Use the regression  expense = a + b*eat_out  to get cost per home meal and per outside meal:
         a = 90*c_home  (expense with zero outside meals),   b = extra cost of one outside meal."""
    b, a = ols_simple(df.eat_out.values.astype(float), df.expense.values.astype(float))
    c1 = float(np.clip(a / MEALS_PER_MONTH, 20, 200))
    c2 = float(c1 + max(b, 5.0))
    return b, a, c1, c2


def design_matrix(df, full):
    cols, names = [np.ones(len(df)), df.eat_out.values.astype(float)], ["intercept", "eat_out"]
    if full:
        for c in ("income", "hostel"):
            if c in df and df[c].notna().all():
                cols.append(df[c].values.astype(float)); names.append(c)
    return np.column_stack(cols), names


# ------------------------------  LPP : simplex, duality, convexity  ------------------------------
def simplex_max(c, A, b, max_iter=200):
    """Tableau simplex for  max c.x  s.t.  A x <= b,  x >= 0,  b >= 0   (origin is the starting vertex)."""
    c, A, b = np.asarray(c, float), np.asarray(A, float), np.asarray(b, float)
    m, n = A.shape
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n], T[:m, n:n + m], T[:m, -1], T[m, :n] = A, np.eye(m), b, -c
    basis = list(range(n, n + m))
    log = [dict(it=0, enter=None, leave=None, T=T.copy(), basis=basis.copy())]
    status = "max_iter"
    for it in range(1, max_iter + 1):
        row = T[m, :-1]
        j = int(np.argmin(row)) if it <= 50 else int(np.where(row < -1e-9)[0][0] if (row < -1e-9).any() else 0)
        if row[j] >= -1e-9:
            status = "optimal"; break
        col = T[:m, j]
        pos = col > 1e-9
        if not pos.any():
            status = "unbounded"; break
        ratios = np.full(m, np.inf); ratios[pos] = T[:m, -1][pos] / col[pos]
        i = int(np.argmin(ratios))
        T[i] /= T[i, j]
        for r in range(m + 1):
            if r != i:
                T[r] -= T[r, j] * T[i]
        leaving = basis[i]; basis[i] = j
        log.append(dict(it=it, enter=j, leave=leaving, T=T.copy(), basis=basis.copy()))
    x = np.zeros(n + m)
    for i, bv in enumerate(basis):
        x[bv] = T[i, -1]
    return dict(status=status, x=x[:n], z=T[m, -1], T=T, basis=basis, log=log, n=n, m=m)


def solve_min_via_dual(c, G, h):
    """PRIMAL:  min c.x  s.t. G x >= h, x >= 0      (c >= 0)
       DUAL  :  max h.y  s.t. G^T y <= c, y >= 0    <- origin feasible, so plain simplex applies.
       Primal solution x* is read from the objective row under the slack columns (strong duality)."""
    c, G, h = np.asarray(c, float), np.asarray(G, float), np.asarray(h, float)
    k, p = G.shape
    res = simplex_max(h, G.T, c)
    out = dict(res=res, k=k, p=p)
    if res["status"] != "optimal":
        out["status"] = "primal infeasible" if res["status"] == "unbounded" else res["status"]
        return out
    zrow = res["T"][-1]
    out.update(status="optimal", y=res["x"][:k], x=zrow[k:k + p], surplus=zrow[:k], dual_obj=res["z"])
    out["cost"] = float(c @ out["x"])
    return out


def feasible_vertices(G, h):
    """Extreme points of {x : G x >= h, x >= 0} in R^2 : intersect every pair of boundary lines, keep feasible ones."""
    A = np.vstack([G, np.eye(2)])
    b = np.concatenate([h, [0, 0]])
    pts = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            M = A[[i, j]]
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            x = np.linalg.solve(M, b[[i, j]])
            if np.all(G @ x >= h - 1e-7) and np.all(x >= -1e-7):
                if not any(np.allclose(x, q, atol=1e-6) for q in pts):
                    pts.append(x)
    return np.array(pts)


def fmt_tableau(entry, names):
    T, basis = entry["T"], entry["basis"]
    hdr = f"{'Basis':>6}" + "".join(f"{n:>10}" for n in names) + f"{'RHS':>10}"
    lines = [hdr]
    for i, bv in enumerate(basis):
        lines.append(f"{names[bv]:>6}" + "".join(f"{v:10.3f}" for v in T[i, :-1]) + f"{T[i, -1]:10.2f}")
    lines.append(f"{'z':>6}" + "".join(f"{v:10.3f}" for v in T[-1, :-1]) + f"{T[-1, -1]:10.2f}")
    return "\n".join(lines)


# =====================================================================================
#  PART B : GUI
# =====================================================================================
class BaseTab(ttk.Frame):
    def __init__(self, master, app, figsize=(8, 6), text_w=62):
        super().__init__(master)
        self.app = app
        self.columnconfigure(1, weight=1); self.rowconfigure(0, weight=1)
        left = ttk.Frame(self); left.grid(row=0, column=0, sticky="ns", padx=6, pady=6)
        self.ctrl = ttk.LabelFrame(left, text="Controls"); self.ctrl.pack(fill="x")
        tf = ttk.Frame(left); tf.pack(fill="both", expand=True, pady=(6, 0))
        self.text = tk.Text(tf, width=text_w, height=10, font=("Consolas", 9), wrap="none")
        ys = ttk.Scrollbar(tf, orient="vertical", command=self.text.yview)
        xs = ttk.Scrollbar(tf, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        xs.pack(side="bottom", fill="x"); ys.pack(side="right", fill="y"); self.text.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(self); right.grid(row=0, column=1, sticky="nsew")
        self.fig = Figure(figsize=figsize, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.canvas, right).update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._r = 0

    def row(self, label, widget_factory):
        ttk.Label(self.ctrl, text=label, wraplength=230).grid(row=self._r, column=0, sticky="w", padx=4, pady=2)
        w = widget_factory(self.ctrl); w.grid(row=self._r, column=1, sticky="ew", padx=4); self._r += 1
        return w

    def button(self, label, cmd):
        b = ttk.Button(self.ctrl, text=label, command=cmd)
        b.grid(row=self._r, column=0, columnspan=2, sticky="ew", padx=4, pady=2); self._r += 1
        return b

    def say(self, s):
        self.text.delete("1.0", "end"); self.text.insert("1.0", s)

    def draw(self):
        self.fig.tight_layout(); self.canvas.draw_idle()

    def refresh(self):
        pass


# ------------------------------------------ TAB 1 ------------------------------------------
class ColumnMapper(tk.Toplevel):
    def __init__(self, master, cols):
        super().__init__(master); self.title("Map your Google-Form columns"); self.result = None
        opts = ["(none)"] + list(cols); self.vars = {}
        for i, (key, lab) in enumerate([("eat_out", "Times eating outside per month  *"),
                                        ("expense", "Monthly food expense (Rs)  *"),
                                        ("income", "Monthly income / allowance (optional)"),
                                        ("hostel", "Hostel / PG ? Yes-No (optional)")]):
            ttk.Label(self, text=lab).grid(row=i, column=0, sticky="w", padx=8, pady=5)
            v = tk.StringVar(value="(none)"); self.vars[key] = v
            ttk.Combobox(self, textvariable=v, values=opts, width=45, state="readonly").grid(row=i, column=1, padx=8)
        ttk.Button(self, text="OK", command=self.ok).grid(row=5, column=0, columnspan=2, pady=8)
        self.grab_set()

    def ok(self):
        self.result = {k: v.get() for k, v in self.vars.items()}; self.destroy()


class DataTab(BaseTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.v = {}
        for lab, key, val in [("Respondents (n)", "n", 200), ("Random seed", "seed", 42),
                              ("Base food cost / month (Rs)", "base", 4500), ("Income effect (Rs per Rs)", "inc", 0.03),
                              ("Extra cost per outside meal (Rs)", "cost", 100), ("Noise SD (Rs)", "noise", 500)]:
            self.v[key] = tk.StringVar(value=str(val))
            self.row(lab, lambda p, k=key: ttk.Entry(p, textvariable=self.v[k], width=10))
        self.button("Simulate new survey", self.simulate)
        self.button("Load real Google-Form CSV / XLSX ...", self.load)

    def simulate(self):
        try:
            df = generate_survey(int(self.v["n"].get()), int(self.v["seed"].get()), float(self.v["base"].get()),
                                 float(self.v["inc"].get()), float(self.v["cost"].get()), float(self.v["noise"].get()))
        except ValueError:
            messagebox.showerror("Input error", "Please enter valid numbers."); return
        self.app.set_data(df, "Simulated survey")

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Survey files", "*.csv *.xlsx"), ("All", "*.*")])
        if not path: return
        try:
            raw = pd.read_excel(path) if path.lower().endswith("xlsx") else pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Read error", str(e)); return
        dlg = ColumnMapper(self, raw.columns); self.wait_window(dlg)
        m = dlg.result
        if not m or m["eat_out"] == "(none)" or m["expense"] == "(none)":
            return
        df = pd.DataFrame({"eat_out": pd.to_numeric(raw[m["eat_out"]], errors="coerce"),
                           "expense": pd.to_numeric(raw[m["expense"]], errors="coerce")})
        df["income"] = pd.to_numeric(raw[m["income"]], errors="coerce") if m["income"] != "(none)" else np.nan
        if m["hostel"] != "(none)":
            df["hostel"] = raw[m["hostel"]].astype(str).str.strip().str.lower().isin(["1", "yes", "y", "true", "hostel", "pg"]).astype(float)
        else:
            df["hostel"] = np.nan
        df = df.dropna(subset=["eat_out", "expense"]).reset_index(drop=True)
        if len(df) < 10:
            messagebox.showerror("Too few rows", "Need at least 10 valid rows (numeric columns)."); return
        df.insert(0, "id", np.arange(1, len(df) + 1))
        self.app.set_data(df, "Loaded: " + path.split("/")[-1])

    def refresh(self):
        df = self.app.df
        self.say(f"SOURCE : {self.app.source}\nROWS   : {len(df)}\n\n" +
                 df[["eat_out", "expense", "income", "hostel"]].describe().round(2).to_string() +
                 "\n\nFirst rows:\n" + df.head(12).to_string(index=False) +
                 "\n\nHOW THE SIMULATION WORKS\n  eat_out ~ Poisson(lambda_i), lambda_i grows with income & hostel life\n"
                 "  expense = base + a*income + cost*eat_out + Normal(0, noise)\n"
                 "  -> a true linear relation + randomness, so every later tab has something real to find.")
        self.fig.clear()
        a = self.fig.subplots(2, 2)
        a[0, 0].hist(df.eat_out, bins=min(30, max(5, int(df.eat_out.max()))), color="tab:blue", edgecolor="w")
        a[0, 0].set(title="Times eating outside / month", xlabel="eat_out", ylabel="students")
        a[0, 1].hist(df.expense, bins=25, color="tab:orange", edgecolor="w")
        a[0, 1].set(title="Monthly food expense", xlabel="Rs")
        a[1, 0].scatter(df.eat_out, df.expense, s=12, alpha=.6)
        a[1, 0].set(title="Eating out vs expense", xlabel="eat_out", ylabel="Rs")
        g = pd.qcut(df.eat_out.rank(method="first"), 3, labels=["Low", "Medium", "High"])
        a[1, 1].boxplot([df.expense[g == l] for l in ["Low", "Medium", "High"]])
        a[1, 1].set_xticks([1, 2, 3]); a[1, 1].set_xticklabels(["Low", "Medium", "High"])
        a[1, 1].set(title="Expense by eating-out tercile", ylabel="Rs")
        self.draw()


# ------------------------------------------ TAB 2 ------------------------------------------
class ProbTab(BaseTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.kv, self.tv = tk.DoubleVar(), tk.DoubleVar()
        self.sk = self.row("F = 'frequent eater':  eats out >= k times", lambda p: tk.Scale(p, variable=self.kv, orient="horizontal", length=170, resolution=1, command=lambda _: self.update_plot()))
        self.st = self.row("H = 'high expense':  expense >= Rs t", lambda p: tk.Scale(p, variable=self.tv, orient="horizontal", length=170, resolution=50, command=lambda _: self.update_plot()))

    def refresh(self):
        df = self.app.df
        self.sk.configure(from_=0, to=int(df.eat_out.max())); self.st.configure(from_=int(df.expense.min()), to=int(df.expense.max()))
        self.kv.set(float(np.median(df.eat_out))); self.tv.set(float(np.median(df.expense))); self.update_plot()

    def update_plot(self):
        df = self.app.df; k, t = self.kv.get(), self.tv.get()
        r = prob_table(df, k, t)
        f = lambda v: f"{v:.4f}"
        self.say(
            f"EVENTS   F = eat_out >= {k:.0f}      H = expense >= Rs {t:.0f}\n\n"
            f"CONTINGENCY TABLE (counts)\n              H      not H     total\n"
            f"  F      {r['nFH']:5d}   {r['nF']-r['nFH']:5d}     {r['nF']:5d}\n"
            f"  not F  {r['nFcH']:5d}   {r['n']-r['nF']-r['nFcH']:5d}     {r['n']-r['nF']:5d}\n"
            f"  total  {r['nH']:5d}   {r['n']-r['nH']:5d}     {r['n']:5d}\n\n"
            f"P(F)            = {f(r['pF'])}\nP(H)            = {f(r['pH'])}\nP(F and H)      = {f(r['pFH'])}\n"
            f"P(H | F)        = P(F&H)/P(F)   = {f(r['pH_F'])}\nP(H | not F)    = {f(r['pH_Fc'])}\n\n"
            f"TOTAL PROBABILITY\n  P(H)=P(H|F)P(F)+P(H|Fc)P(Fc) = {f(r['pH_total'])}   (direct: {f(r['pH'])})\n\n"
            f"BAYES THEOREM\n  P(F|H) = P(H|F)P(F) / P(H) = {f(r['pF_H_bayes'])}\n  direct count            = {f(r['pF_H_direct'])}\n\n"
            f"INDEPENDENCE TEST\n  P(F)P(H) = {f(r['pF']*r['pH'])}  vs  P(F&H) = {f(r['pFH'])}\n"
            f"  {'=> NOT independent: eating out and high expense are dependent' if abs(r['pF']*r['pH']-r['pFH'])>0.02 else '=> approximately independent'}\n"
            f"  Lift P(H|F)/P(H) = {r['pH_F']/r['pH'] if r['pH'] else float('nan'):.2f}")
        self.fig.clear(); a = self.fig.subplots(1, 2)
        labs = ["P(F)", "P(H)", "P(H|F)", "P(H|Fc)", "P(F|H)"]
        vals = [r["pF"], r["pH"], r["pH_F"], r["pH_Fc"], r["pF_H_bayes"]]
        a[0].bar(labs, vals, color=["#999", "#999", "tab:red", "tab:green", "tab:purple"])
        for i, v in enumerate(vals): a[0].text(i, v + .01, f"{v:.2f}", ha="center")
        a[0].set(ylim=(0, 1.1), title="Probabilities")
        ks = np.arange(0, int(df.eat_out.max()) + 1)
        H = r["H"]
        curve = [H[df.eat_out.values >= q].mean() if (df.eat_out.values >= q).any() else np.nan for q in ks]
        a[1].plot(ks, curve, marker="o", ms=3); a[1].axvline(k, color="r", ls="--")
        a[1].set(title="P(high expense | eats out >= k)", xlabel="k", ylabel="probability")
        self.draw()


# ------------------------------------------ TAB 3 ------------------------------------------
class DistTab(BaseTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.tv = tk.DoubleVar(value=0.2)
        self.row("MGF plot range  |t| <=", lambda p: tk.Scale(p, variable=self.tv, from_=0.05, to=0.4, resolution=0.05, orient="horizontal", length=170, command=lambda _: self.refresh()))

    def refresh(self):
        x = self.app.df.eat_out.values.astype(float); n = len(x)
        mo = moments(x); lam = mo["mean"]; p = lam / MEALS_PER_MONTH
        ks = np.arange(0, int(x.max()) + 6)
        emp = np.bincount(x.astype(int), minlength=len(ks))[:len(ks)] / n
        bp, pp = binom_pmf(ks, MEALS_PER_MONTH, p), poisson_pmf(ks, lam)
        h = 1e-3
        M = lambda t: np.mean(np.exp(t * x))
        d1 = (M(h) - M(-h)) / (2 * h); d2 = (M(h) - 2 * M(0) + M(-h)) / h ** 2
        wk = [1 - sum(binom_pmf(np.arange(r), 21, p)) for r in (3, 5, 7, 9)]
        self.say(
            f"X = outside meals in a month (of {MEALS_PER_MONTH})\n"
            f"Bernoulli model per meal: p-hat = mean/90 = {p:.4f}\n"
            f"  X ~ Binomial(90, p):  mean {MEALS_PER_MONTH*p:.2f}  var {MEALS_PER_MONTH*p*(1-p):.2f}\n"
            f"  X ~ Poisson(lam={lam:.2f}):  mean {lam:.2f}  var {lam:.2f}\n"
            f"  DATA            :  mean {mo['mean']:.2f}  var {mo['var']:.2f}\n"
            f"  dispersion var/mean = {mo['var']/mo['mean']:.2f}  ({'over-dispersed: students differ (mixture)' if mo['var']/mo['mean']>1.3 else 'close to Poisson'})\n\n"
            f"MOMENTS\n  raw     E[X^k]      : " + "  ".join(f"{v:10.2f}" for v in mo['raw']) +
            f"\n  central E[(X-m)^k]  : " + "  ".join(f"{v:10.2f}" for v in mo['cen']) +
            f"\n  skewness = {mo['skew']:.3f}   excess kurtosis = {mo['kurt']:.3f}\n\n"
            f"MGF  M(t)=E[e^(tX)]\n  M'(0)  numerical = {d1:.3f}   (= E[X]   = {mo['raw'][0]:.3f})\n"
            f"  M''(0) numerical = {d2:.2f}  (= E[X^2] = {mo['raw'][1]:.2f})\n"
            f"  Poisson MGF  exp(lam(e^t-1)),  Binomial MGF (1-p+p e^t)^90\n\n"
            f"WEEKLY BINOMIAL  (21 meals, p={p:.3f})\n  P(>=3)={wk[0]:.3f}  P(>=5)={wk[1]:.3f}  P(>=7)={wk[2]:.3f}  P(>=9)={wk[3]:.3f}")
        self.fig.clear(); a = self.fig.subplots(1, 2)
        a[0].bar(ks, emp, color="lightgray", edgecolor="k", label="data")
        a[0].plot(ks, bp, "o-", ms=3, label="Binomial(90,p)"); a[0].plot(ks, pp, "s--", ms=3, label="Poisson(lam)")
        a[0].set(title="Empirical PMF vs theory", xlabel="outside meals / month", ylabel="P(X=k)"); a[0].legend()
        ts = np.linspace(-self.tv.get(), self.tv.get(), 120)
        a[1].plot(ts, [math.log(M(t)) for t in ts], lw=3, color="lightgray", label="data  ln M(t)")
        a[1].plot(ts, lam * (np.exp(ts) - 1), "--", label="Poisson")
        a[1].plot(ts, MEALS_PER_MONTH * np.log(1 - p + p * np.exp(ts)), ":", label="Binomial")
        a[1].set(title="log-MGF: slope at 0 = mean", xlabel="t"); a[1].legend()
        self.draw()


# ------------------------------------------ TAB 4 ------------------------------------------
class CLTTab(BaseTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.var = tk.StringVar(value="eat_out"); self.nv = tk.IntVar(value=5); self.bud = tk.StringVar(value="7000"); self.seed = 1
        self.row("Variable", lambda p: ttk.Combobox(p, textvariable=self.var, values=["eat_out", "expense"], state="readonly", width=10))
        self.var.trace_add("write", lambda *_: self.refresh())
        self.row("Sample size n", lambda p: tk.Scale(p, variable=self.nv, from_=1, to=200, orient="horizontal", length=170, command=lambda _: self.refresh()))
        self.row("Budget Rs (for expense)", lambda p: ttk.Entry(p, textvariable=self.bud, width=10))
        self.button("Re-draw 3000 new samples", self.resample)
        self.button("Compute probabilities", self.refresh)

    def resample(self):
        self.seed += 1; self.refresh()

    def refresh(self):
        df = self.app.df; v = self.var.get(); x = df[v].values.astype(float)
        n = self.nv.get(); mu, sd = x.mean(), x.std(ddof=0)
        rng = np.random.default_rng(self.seed)
        means = rng.choice(x, size=(3000, n), replace=True).mean(axis=1)
        se = sd / math.sqrt(n)
        try: B = float(self.bud.get())
        except ValueError: B = float(np.median(df.expense))
        z = (B - df.expense.mean()) / df.expense.std(ddof=0)
        ci = 1.96 * x.std(ddof=1) / math.sqrt(len(x))
        sk0, sk1 = moments(x)["skew"], moments(means)["skew"]
        self.say(
            f"POPULATION = the {len(x)} survey answers of '{v}'\n  mu = {mu:.2f}   sigma = {sd:.2f}   skewness = {sk0:.3f}\n\n"
            f"CLT EXPERIMENT: 3000 samples of size n = {n}\n  mean of sample means = {means.mean():.2f}  (theory mu = {mu:.2f})\n"
            f"  SD of sample means   = {means.std():.3f}  (theory sigma/sqrt(n) = {se:.3f})\n"
            f"  skewness of means    = {sk1:.3f}  (population {sk0:.3f}  ->  moves to 0 as n grows)\n\n"
            f"95% CONFIDENCE INTERVAL for mean '{v}' (whole survey)\n  {mu-ci:.2f}  to  {mu+ci:.2f}\n\n"
            f"NORMAL MODEL for expense  (mu={df.expense.mean():.0f}, sigma={df.expense.std(ddof=0):.0f})\n  Budget Rs {B:.0f}:  z = {z:.3f}\n"
            f"  P(expense <= budget) normal = {normal_cdf(B, df.expense.mean(), df.expense.std(ddof=0)):.4f}   empirical = {np.mean(df.expense <= B):.4f}\n"
            f"  Group of {n}: P(avg expense <= budget) = {normal_cdf(B, df.expense.mean(), df.expense.std(ddof=0)/math.sqrt(n)):.4f}")
        self.fig.clear(); a = self.fig.subplots(1, 2)
        g = np.linspace(x.min() - sd, x.max() + sd, 300)
        a[0].hist(x, bins=25, density=True, color="lightgray", edgecolor="w")
        a[0].plot(g, normal_pdf(g, mu, sd), "r", label="Normal(mu,sigma)")
        a[0].set(title=f"Population '{v}'"); a[0].legend()
        g2 = np.linspace(means.min(), means.max(), 300)
        a[1].hist(means, bins=40, density=True, color="tab:blue", edgecolor="w", alpha=.8)
        a[1].plot(g2, normal_pdf(g2, mu, se), "r", label="Normal(mu, sigma/sqrt n)")
        a[1].set(title=f"Distribution of sample means (n = {n})"); a[1].legend()
        self.draw()


# ------------------------------------------ TAB 5 ------------------------------------------
class CorrTab(BaseTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.out = tk.BooleanVar(value=False)
        self.row("Add 3 outliers (rank correlation is robust)", lambda p: ttk.Checkbutton(p, variable=self.out, command=self.refresh))
        self.button("Re-run permutation test", self.refresh)

    def refresh(self):
        df = self.app.df; x = df.eat_out.values.astype(float); y = df.expense.values.astype(float)
        if self.out.get():
            x = np.append(x, [x.min()] * 3); y = np.append(y, [y.max() * 2.5] * 3)
        n = len(x)
        rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
        r = np.corrcoef(x, y)[0, 1]; rho = np.corrcoef(rx, ry)[0, 1]
        cov = np.cov(x, y)[0, 1]
        rng = np.random.default_rng()
        B = 2000; rp, sp = np.empty(B), np.empty(B)
        for i in range(B):
            idx = rng.permutation(n); rp[i] = np.corrcoef(x, y[idx])[0, 1]; sp[i] = np.corrcoef(rx, ry[idx])[0, 1]
        p_r = (np.sum(np.abs(rp) >= abs(r)) + 1) / (B + 1); p_s = (np.sum(np.abs(sp) >= abs(rho)) + 1) / (B + 1)
        zf = math.atanh(r); lo, hi = math.tanh(zf - 1.96 / math.sqrt(n - 3)), math.tanh(zf + 1.96 / math.sqrt(n - 3))
        self.say(
            f"n = {n}{'  (with 3 injected outliers)' if self.out.get() else ''}\n\n"
            f"Covariance        = {cov:,.2f}\nPearson r         = {r:.4f}      r^2 = {r*r:.4f}\n"
            f"  95% CI (Fisher) = [{lo:.3f}, {hi:.3f}]\nSpearman rho      = {rho:.4f}   (Pearson r on ranks)\n\n"
            f"PERMUTATION TEST (2000 shuffles of expense)\n  H0: eating out and expense unrelated\n"
            f"  p-value (Pearson)  = {p_r:.4f}\n  p-value (Spearman) = {p_s:.4f}\n\n"
            f"Reading: |r| near 1 => strong LINEAR link; rho measures any MONOTONE link and\nbarely moves when outliers appear, r can change a lot.\n"
            f"Correlation is not causation: income drives both eating out and expense (confounder).")
        self.fig.clear(); a = self.fig.subplots(1, 3)
        a[0].scatter(x, y, s=12, alpha=.6); a[0].set(title=f"Raw data   r={r:.3f}", xlabel="eat_out", ylabel="expense")
        a[1].scatter(rx, ry, s=12, alpha=.6, color="tab:green"); a[1].set(title=f"Ranks   rho={rho:.3f}", xlabel="rank(eat_out)", ylabel="rank(expense)")
        a[2].hist(rp, bins=40, color="lightgray", edgecolor="w"); a[2].axvline(r, color="r"); a[2].axvline(-r, color="r")
        a[2].set(title="Null distribution of r (shuffled)")
        self.draw()


# ------------------------------------------ TAB 6 ------------------------------------------
class RegTab(BaseTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.pv = tk.StringVar(value="20")
        self.row("Predict expense for eat_out =", lambda p: ttk.Entry(p, textvariable=self.pv, width=8))
        self.button("Update prediction", self.refresh)

    def refresh(self):
        df = self.app.df; y = df.expense.values.astype(float); n = len(y)
        x = df.eat_out.values.astype(float)
        b1, b0 = ols_simple(x, y)
        X, names = design_matrix(df, True); p = X.shape[1]
        XtX = X.T @ X
        beta = np.linalg.solve(XtX, X.T @ y)                    # normal equations  (X^T X) b = X^T y
        yhat = X @ beta; e = y - yhat
        sse, sst = e @ e, np.sum((y - y.mean()) ** 2)
        r2 = 1 - sse / sst; ar2 = 1 - (sse / (n - p)) / (sst / (n - 1))
        s2 = sse / (n - p); se = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
        rank = np.linalg.matrix_rank(X)
        Q, R = np.linalg.qr(X)                                   # Q = orthonormal BASIS of column space (Gram-Schmidt)
        H = X @ np.linalg.inv(XtX) @ X.T
        r2s = 1 - np.sum((y - (b0 + b1 * x)) ** 2) / sst
        try: pv = float(self.pv.get())
        except ValueError: pv = 10.0
        self.say(
            f"SIMPLE MODEL   expense = b0 + b1*eat_out   (Lec 16, least squares)\n  b1 = Sxy/Sxx = {b1:.2f} Rs per outside meal\n  b0 = ybar - b1*xbar = {b0:.1f}   R^2 = {r2s:.4f}\n"
            f"  prediction at eat_out={pv:.0f}  ->  Rs {b0+b1*pv:,.0f}\n\n"
            f"MULTIPLE MODEL  y = X*beta,  columns of X = {names}\n  Normal equations  (X^T X) beta = X^T y\n" +
            "".join(f"  {nm:10s} = {b:12.4f}   (SE {s:.4f}, t = {b/s:.1f})\n" for nm, b, s in zip(names, beta, se)) +
            f"  R^2 = {r2:.4f}   adjusted R^2 = {ar2:.4f}   s = {math.sqrt(s2):.1f}\n\n"
            f"VECTOR-SPACE VIEW  (Lec 19-22)\n  y lives in R^{n};  Col(X) is a SUBSPACE of R^{n}\n"
            f"  dim Col(X) = rank(X) = {rank}  -> basis = the {p} columns (QR gives an orthonormal basis, ||Q'Q-I|| = {np.linalg.norm(Q.T@Q-np.eye(p)):.1e})\n"
            f"  Linear transformation T: R^{p} -> R^{n}, T(beta)=X beta\n  Rank-Nullity: {p} = rank {rank} + nullity {p-rank}  (nullity 0 => T one-to-one)\n"
            f"  Fitted y-hat = H y,  H = projection onto Col(X)\n    trace(H) = {np.trace(H):.3f} = rank,   ||H^2-H|| = {np.linalg.norm(H@H-H):.1e} (idempotent)\n"
            f"    ||X^T e|| = {np.linalg.norm(X.T@e):.1e}  -> residual is ORTHOGONAL to Col(X)\n"
            f"  cond(X^T X) = {np.linalg.cond(XtX):.1e}  (large => standardise columns)")
        self.fig.clear(); a = self.fig.subplots(2, 2)
        xs = np.linspace(x.min(), x.max(), 100)
        a[0, 0].scatter(x, y, s=10, alpha=.6); a[0, 0].plot(xs, b0 + b1 * xs, "r", lw=2)
        a[0, 0].set(title=f"Least-squares line (slope {b1:.1f})", xlabel="eat_out", ylabel="expense")
        a[0, 1].scatter(yhat, e, s=10, alpha=.6); a[0, 1].axhline(0, color="r")
        a[0, 1].set(title="Residuals vs fitted (multiple model)", xlabel="fitted")
        g = np.linspace(e.min(), e.max(), 200)
        a[1, 0].hist(e, bins=25, density=True, color="lightgray", edgecolor="w"); a[1, 0].plot(g, normal_pdf(g, 0, e.std()), "r")
        a[1, 0].set(title="Residuals ~ Normal?")
        a[1, 1].scatter(yhat, y, s=10, alpha=.6); lim = [y.min(), y.max()]; a[1, 1].plot(lim, lim, "r")
        a[1, 1].set(title=f"Actual vs predicted  (R^2={r2:.3f})", xlabel="predicted", ylabel="actual")
        self.draw()


# ------------------------------------------ TAB 7 ------------------------------------------
class LPTab(BaseTab):
    def __init__(self, master, app):
        super().__init__(master, app, text_w=70)
        self.v = {}
        defaults = [("c1  cost / home meal (Rs)", "c1", 58), ("c2  cost / outside meal (Rs)", "c2", 158),
                    ("M   meals needed per month", "M", 90), ("p1  protein / home meal (g)", "p1", 20),
                    ("p2  protein / outside meal (g)", "p2", 12), ("P   protein needed per month (g)", "P", 1720),
                    ("S   min social outside meals", "S", 12), ("Hmax  home-cooking capacity (meals)", "H", 80)]
        for lab, key, val in defaults:
            self.v[key] = tk.StringVar(value=str(val))
            self.row(lab, lambda p, k=key: ttk.Entry(p, textvariable=self.v[k], width=9))
        self.tab = tk.BooleanVar(value=False)
        self.row("Print simplex tableaus", lambda p: ttk.Checkbutton(p, variable=self.tab, command=self.solve))
        self.button("SOLVE  (Simplex + Duality + Convex set)", self.solve)
        self.button("Sensitivity: vary restaurant price c2", self.sens)
        self.button("Monte-Carlo: random prices (500 runs)", self.monte)
        self.ax2_mode = "sens"

    def params(self):
        try: return {k: float(v.get()) for k, v in self.v.items()}
        except ValueError:
            messagebox.showerror("Input error", "Enter valid numbers."); return None

    def model(self, q):
        G = np.array([[1, 1], [q["p1"], q["p2"]], [0, 1], [-1, 0]], float)
        h = np.array([q["M"], q["P"], q["S"], -q["H"]], float)
        labels = ["meals     x1 + x2 >= M", "protein   p1 x1 + p2 x2 >= P", "social    x2 >= S", "capacity  -x1 >= -Hmax"]
        return np.array([q["c1"], q["c2"]], float), G, h, labels

    def refresh(self):
        b, a, c1, c2 = estimate_prices(self.app.df)
        self.v["c1"].set(f"{c1:.1f}"); self.v["c2"].set(f"{c2:.1f}")
        self.solve()

    def solve(self):
        q = self.params()
        if q is None: return
        c, G, h, labels = self.model(q)
        sol = solve_min_via_dual(c, G, h)
        V = feasible_vertices(G, h)
        out = ["LPP  - cheapest way to eat for one month (x1 = home/mess meals, x2 = outside meals)\n",
               "  PRIMAL   min  Z = c1 x1 + c2 x2 = %.1f x1 + %.1f x2" % (c[0], c[1]), "           subject to:"]
        out += [f"    ({i+1}) {l}   [{G[i,0]:g} x1 + {G[i,1]:g} x2 >= {h[i]:g}]" for i, l in enumerate(labels)]
        out += ["           x1, x2 >= 0\n",
                "  DUAL     max  W = " + " + ".join(f"({h[i]:g}) y{i+1}" for i in range(len(h))),
                "           s.t.  G^T y <= c :"]
        out += [f"    {G[0,j]:g} y1 + {G[1,j]:g} y2 + {G[2,j]:g} y3 + {G[3,j]:g} y4 <= {c[j]:g}   (for x{j+1})" for j in range(2)]
        out += ["           y >= 0   (origin is feasible -> simplex starts at y = 0)\n"]
        names = [f"y{i+1}" for i in range(4)] + ["s1", "s2"]
        if self.tab.get():
            out.append("SIMPLEX ON THE DUAL")
            for e in sol["res"]["log"]:
                head = "Initial tableau" if e["enter"] is None else f"Iteration {e['it']}: {names[e['enter']]} enters, {names[e['leave']]} leaves"
                out += [head, fmt_tableau(e, names), ""]
        else:
            out.append(f"Simplex on the dual finished in {len(sol['res']['log'])-1} iterations (tick the box to see tableaus).\n")
        if sol["status"] != "optimal":
            out.append(f"RESULT: {sol['status'].upper()}  - constraints contradict each other (e.g. capacity too small for protein).")
            self.say("\n".join(out)); self.sol_ok = False
            self.plot_region(c, G, h, labels, None, V); return
        x, y_ = sol["x"], sol["y"]
        slack = G @ x - h
        out += ["RESULT",
                f"  Optimal primal  x1* = {x[0]:.3f} home meals,  x2* = {x[1]:.3f} outside meals",
                f"  Minimum monthly food cost Z* = Rs {sol['cost']:,.2f}",
                f"  Optimal dual    y* = ({', '.join(f'{v:.3f}' for v in y_)})   W* = {sol['dual_obj']:,.2f}",
                f"  STRONG DUALITY: Z* = W*  ->  {'HOLDS' if abs(sol['cost']-sol['dual_obj'])<1e-6 else 'violated'}\n",
                "SHADOW PRICES (dual values) & COMPLEMENTARY SLACKNESS"]
        for i, l in enumerate(labels):
            out.append(f"  y{i+1}={y_[i]:8.3f}  slack={slack[i]:8.3f}  {'BINDING' if slack[i]<1e-6 else 'not binding'}   <- {l}")
        out.append(f"  y_i * slack_i = {np.round(y_*slack, 8)}  (all 0)")
        out.append("  Meaning: relaxing a binding constraint by 1 unit changes the cost by y_i rupees.\n")
        vc = np.array([c @ v for v in V])
        out += ["CONVEX SET / EXTREME POINTS", f"  Feasible region = intersection of half-planes = CONVEX polyhedron with {len(V)} corner points:"]
        out += [f"    ({v[0]:7.2f}, {v[1]:7.2f})   Z = {z:10.2f}" for v, z in zip(V, vc)]
        out.append(f"  Best corner Z = {vc.min():.2f}  vs simplex {sol['cost']:.2f}  -> optimum lies at a corner (fundamental theorem of LP)")
        self.sol_ok = True; self.sol = sol
        self.say("\n".join(out))
        self.plot_region(c, G, h, labels, x, V)

    def plot_region(self, c, G, h, labels, xopt, V):
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(1, 2, 1); self.ax2 = self.fig.add_subplot(1, 2, 2)
        a = self.ax1
        xm = max(float(V[:, 0].max()) * 1.5 if len(V) else 0, 100); ym = max(float(V[:, 1].max()) * 1.5 if len(V) else 0, 100)
        xs, ys = np.linspace(0, xm, 400), np.linspace(0, ym, 400); XX, YY = np.meshgrid(xs, ys)
        feas = np.ones_like(XX, bool)
        for i in range(len(h)): feas &= G[i, 0] * XX + G[i, 1] * YY >= h[i] - 1e-9
        if feas.any(): a.contourf(XX, YY, feas.astype(float), levels=[.5, 1.5], colors=["tab:green"], alpha=.25)
        cols = ["tab:blue", "tab:orange", "tab:green", "tab:brown"]
        for i in range(len(h)):
            if abs(G[i, 1]) > 1e-12: a.plot(xs, (h[i] - G[i, 0] * xs) / G[i, 1], color=cols[i % 4], label=labels[i].split()[0])
            else: a.axvline(h[i] / G[i, 0], color=cols[i % 4], label=labels[i].split()[0])
        if len(V): a.scatter(V[:, 0], V[:, 1], c="k", zorder=5, s=25, label="corner points")
        if xopt is not None:
            z = c @ xopt
            a.plot(xs, (z - c[0] * xs) / c[1], "r--", label=f"iso-cost Z*={z:,.0f}")
            a.scatter(*xopt, marker="*", s=250, c="r", zorder=6, label="optimum")
            pts = np.column_stack([XX[feas], YY[feas]]); rng = np.random.default_rng(0); ok = 0; T = 3000
            for _ in range(T):
                p1, p2 = pts[rng.integers(len(pts))], pts[rng.integers(len(pts))]; lam = rng.random(); m = lam * p1 + (1 - lam) * p2
                ok += np.all(G @ m >= h - 1e-6)
            self.text.insert("end", f"\n  Convexity simulation: {ok}/{T} random combinations lam*a+(1-lam)*b of feasible points stayed feasible.")
        a.set(xlim=(0, xm), ylim=(0, ym), xlabel="x1  home meals", ylabel="x2  outside meals", title="Feasible region (convex) & optimum")
        a.legend(fontsize=7, loc="upper right")
        self.sens(redraw_only=True)

    def sens(self, redraw_only=False):
        q = self.params()
        if q is None: return
        if not redraw_only:
            self.fig.clear(); self.ax1 = None; self.ax2 = self.fig.add_subplot(1, 1, 1)
        c, G, h, _ = self.model(q); mult = np.linspace(0.05, 2.0, 80); costs, x2s = [], []
        for m in mult:
            s = solve_min_via_dual([c[0], c[1] * m], G, h)
            costs.append(s["cost"] if s["status"] == "optimal" else np.nan); x2s.append(s["x"][1] if s["status"] == "optimal" else np.nan)
        ax = self.ax2; ax.clear()
        ax.plot(c[1] * mult, costs, "b-o", ms=3); ax.set(xlabel="restaurant price c2 (Rs)", ylabel="min cost Z* (Rs)", title="Sensitivity: Z*(c2) piecewise-linear & concave\n(slope = x2*, kink = optimal basis changes)", )
        t = ax.twinx(); t.plot(c[1] * mult, x2s, "g--"); t.set_ylabel("x2*  outside meals", color="g")
        self.draw()

    def monte(self):
        q = self.params()
        if q is None: return
        c, G, h, _ = self.model(q); rng = np.random.default_rng(7); costs, shares = [], []
        for _ in range(500):
            cc = [max(1.0, rng.normal(c[0], 0.10 * c[0])), max(1.0, rng.normal(c[1], 0.15 * c[1]))]
            s = solve_min_via_dual(cc, G, h)
            if s["status"] == "optimal": costs.append(s["cost"]); shares.append(s["x"][1] / q["M"])
        if not costs: return
        costs = np.array(costs)
        if getattr(self, "ax1", None) is None:
            self.solve(); 
        ax = self.ax2; ax.clear()
        ax.hist(costs, bins=30, density=True, color="tab:purple", edgecolor="w", alpha=.8)
        g = np.linspace(costs.min(), costs.max(), 200); ax.plot(g, normal_pdf(g, costs.mean(), costs.std()), "r")
        ax.set(title="Monte-Carlo: optimal cost when prices are random", xlabel="Z* (Rs)")
        self.text.insert("end", f"\n\nMONTE-CARLO (500 random price pairs: c1~N(.,10%), c2~N(.,15%))\n  mean Z* = Rs {costs.mean():,.0f}   SD = {costs.std():,.0f}   "
                               f"90% range = [{np.percentile(costs,5):,.0f}, {np.percentile(costs,95):,.0f}]\n  mean outside-meal share = {np.mean(shares):.1%}")
        self.draw()


# ------------------------------------------ APP ------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Eating Outside vs Monthly Food Expense  -  MFAIDS Simulation")
        self.geometry("1400x820")
        self.df, self.source = generate_survey(), "Simulated survey"
        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True)
        self.tabs = []
        for name, cls in [("1 Data & Simulator", DataTab), ("2 Probability & Bayes", ProbTab),
                          ("3 Distributions & MGF", DistTab), ("4 Normal & CLT", CLTTab),
                          ("5 Correlation", CorrTab), ("6 Regression & Linear Algebra", RegTab),
                          ("7 LPP: Simplex, Duality, Convexity", LPTab)]:
            t = cls(nb, self); nb.add(t, text=name); self.tabs.append(t)
        self.refresh_all()

    def set_data(self, df, source):
        self.df, self.source = df, source; self.refresh_all()

    def refresh_all(self):
        for t in self.tabs: t.refresh()


if __name__ == "__main__":
    App().mainloop()