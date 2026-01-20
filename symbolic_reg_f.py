'''
This files performs symbolic regression on hydraulic data from PHE experiments using
the PySR a wrapping of the Julia library SymbolicRegression.jl. It generates a PDF
with multiple candidate equations that balances complexity and predictive performance.
Results are rendered in LaTeX style.

Arthur | arthur-miguel.github.io | 12/22/2025
'''

import pandas as pd
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec
from sympy import simplify, latex as sympy_latex, Float, sympify, Symbol, parse_expr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os
import subprocess
import re
import warnings

warnings.filterwarnings("ignore")

# Input variables and dataset path
datasets = {"full": "data/data_sr.csv"}
INPUTS = ["Re_medio", "Dpin", "Aperto", "geom"] 

# Sets mapping for variable names and latex rendering of discovered equations
VAR_MAP = {
    0: {"name": "Re",      "latex": "Re"},
    1: {"name": "Dpin",    "latex": "Dp_{in}"},
    2: {"name": "A",       "latex": "A"},
    3: {"name": "c_1",     "latex": "c_1"}, 
    4: {"name": "c_2",     "latex": "c_2"},
    5: {"name": "c_3",     "latex": "c_3"},
}

# Free geometric coefficients
PARAM_NAMES = ["c1", "c2", "c3"]
PARAM_LATEX_MAP = {"c1": "c_1", "c2": "c_2", "c3": "c_3"}

# Sets possible input parameters of final equations, including geometric coefficients
template = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["Re", "Dpin", "A", "geom"],
    parameters={"c1": 3, "c2": 3, "c3": 3},
    combine="f(Re, Dpin, A, c1[geom], c2[geom], c3[geom])"
)

# Defines set of possible operators for genetic programming
def make_model():
    return PySRRegressor(
        expression_spec=template,
        niterations=2500, 
        populations=48,
        maxsize=25,
        parsimony=1e-6,
        constraints={'pow': (-1, 2), 'exp': (1), 'log': (4)},
        binary_operators=["+", "-", "*", "/", "pow"],
        unary_operators=["exp", "log"],
        elementwise_loss="loss(x, y) = (x - y)^2",
        progress=True,
        verbosity=1,
        model_selection="best",
        complexity_of_variables=1e-16,
        procs=32
    )

# Helper functions for LaTeX exporting
def get_safe_latex_fallback(clean_str):
    def replace_hash_safe(match):
        try:
            idx = int(match.group(1)) - 1
            if idx in VAR_MAP: return VAR_MAP[idx]["latex"]
            return f"x_{{{match.group(1)}}}"
        except: return "x"

    safe = re.sub(r'#(\d+)', replace_hash_safe, clean_str)
    safe = safe.replace("*", r" \cdot ").replace("exp", r"\exp").replace("log", r"\log")
    return safe

def process_eqn_to_latex(raw_eqn_str):
    if not isinstance(raw_eqn_str, str): return str(raw_eqn_str)

    parts = raw_eqn_str.split(';')
    clean_str = parts[0]
    if "=" in clean_str: clean_str = clean_str.split("=")[1]
    clean_str = clean_str.strip()

    def replace_hash_sympy(match):
        idx = int(match.group(1)) - 1
        if idx in VAR_MAP: return VAR_MAP[idx]["name"]
        return f"x{idx}"

    sympy_ready_str = re.sub(r'#(\d+)', replace_hash_sympy, clean_str)
    sympy_ready_str = sympy_ready_str.replace("^", "**")

    try:
        local_dict = {v["name"]: Symbol(v["name"]) for v in VAR_MAP.values()}
        expr = sympify(sympy_ready_str, locals=local_dict)
        expr = simplify(expr)
        expr = expr.xreplace({n: Float(round(float(n), 4)) for n in expr.atoms(Float)})
        
        latex_str = sympy_latex(expr)
        
        for idx in sorted(VAR_MAP.keys(), reverse=True):
            name = VAR_MAP[idx]["name"]
            latex_val = VAR_MAP[idx]["latex"]
            latex_str = re.sub(r"\\operatorname\{" + name + r"\}", latex_val, latex_str)
            latex_str = re.sub(r"\b" + name + r"\b", latex_val, latex_str)
            
        return latex_str

    except Exception:
        return get_safe_latex_fallback(clean_str)

def format_params_latex(params):
    items = []
    for name in PARAM_NAMES:
        arr = params.get(name)
        if arr is not None:
            vals = ", ".join([f"{v:.4g}" for v in arr])
            latex_name = PARAM_LATEX_MAP.get(name, name)
            items.append(rf"{latex_name}=[{vals}]")
    
    if not items: return ""
    full_str = ", ".join(items)
    return rf"\mathit{{ {full_str} }}"

class PythonPredictor:
    '''
    Converts Julia expressions from PySR backend to Python expressions for evaluation
    '''
    @staticmethod
    def parse_params(eqn_str):
        params = {}
        for p_name in PARAM_NAMES:
            pattern = p_name + r"\s*=\s*\[(.*?)\]"
            match = re.search(pattern, eqn_str)
            if match:
                vals_str = match.group(1).split(',')
                try: params[p_name] = np.array([float(x) for x in vals_str])
                except: params[p_name] = None
            else: params[p_name] = None
        return params

    @staticmethod
    def predict(eqn_str, X, geom_col_idx=3):
        clean = eqn_str.split(';')[0].split('=')[-1].strip().replace("^", "**")
        def replace_hash(m): return f"x{int(m.group(1))-1}"
        formula_safe = re.sub(r'#(\d+)', replace_hash, clean)
        
        params = PythonPredictor.parse_params(eqn_str)
        geoms = X[:, geom_col_idx].astype(int)
        if geoms.min() >= 1: geoms -= 1
        
        args = [X[:, 0], X[:, 1], X[:, 2]]
        for p_name in PARAM_NAMES:
            if params[p_name] is not None:
                valid_geoms = np.clip(geoms, 0, len(params[p_name]) - 1)
                args.append(params[p_name][valid_geoms])
            else:
                args.append(np.zeros_like(geoms, dtype=float))

        try:
            sym_vars = [Symbol(f"x{i}") for i in range(len(args))]
            sympy_expr = parse_expr(formula_safe)
            from sympy import lambdify
            func = lambdify(sym_vars, sympy_expr, modules="numpy")
            y_pred = func(*args)
            if np.isscalar(y_pred): y_pred = np.full(len(X), y_pred)
            return y_pred
        except Exception:
            return None


def extract_metrics(model, Xtest, ytest):
    df_eq = model.equations_
    rows = []

    for idx, row in df_eq.iterrows():
        raw_eqn = row.get("equation", str(row.get("sympy_format", "")))
        
        latex_eqn_only = process_eqn_to_latex(raw_eqn)
        latex_params_only = format_params_latex(PythonPredictor.parse_params(raw_eqn))
        
        cell_content = r"$\displaystyle " + latex_eqn_only + r"$"
        
        if latex_params_only:
            cell_content += r" \newline \scriptsize $ " + latex_params_only + r" $"

        ypred = PythonPredictor.predict(raw_eqn, Xtest, geom_col_idx=3)
        
        if ypred is None or np.any(np.isnan(ypred)):
            rmse, mae, mape, r2 = np.nan, np.nan, np.nan, np.nan
        else:
            rmse = np.sqrt(mean_squared_error(ytest, ypred))
            mae = mean_absolute_error(ytest, ypred)
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((ytest - ypred) / ytest)) * 100
                if not np.isfinite(mape): mape = np.nan
            r2 = r2_score(ytest, ypred)

        rows.append([cell_content, rmse, mae, mape, r2])

    return rows

# Create and compiles output .tex file from evaluated Python expresions
def build_latex_doc(tables_data):
    doc = r"""
\documentclass{article}
\usepackage{amsmath, amssymb}
\usepackage{longtable}
\usepackage{pdflscape}
\usepackage[margin=1.0cm]{geometry}
\usepackage{booktabs}
\usepackage{xcolor} 
\begin{document}
"""
    for title, rows in tables_data:
        doc += r"\begin{landscape}" + "\n"
        doc += r"\section*{%s}" % title.replace("_", " ") + "\n"
        doc += r"\small" + "\n"
        doc += r"\renewcommand{\arraystretch}{1.7}" + "\n"
        doc += r"\begin{longtable}{p{16cm}cccc}" + "\n"
        doc += r"\toprule" + "\n"
        doc += r"\textbf{Expression} & \textbf{RMSE} & \textbf{MAE} & \textbf{MAPE} & \textbf{R$^2$} \\ \midrule \endhead" + "\n"
        
        for expr, rmse, mae, mape, r2 in rows:
            doc += rf"{expr} & {rmse:.4f} & {mae:.4f} & {mape:.2f} & {r2:.4f} \\" + "\n"
            
        doc += r"\bottomrule" + "\n"
        doc += r"\end{longtable}" + "\n"
        doc += r"\end{landscape}" + "\n\n"

    doc += r"\end{document}"
    return doc

def compile_pdf(tex_path, out_dir):
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", os.path.basename(tex_path)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=out_dir)
        print(f"  [OK] PDF generated: {tex_path.replace('.tex', '.pdf')}")
    except Exception as e:
        print(f"  [Error] PDF compilation failed.")

for tag, path in datasets.items():
    print(f"Processing: {tag}")
    OUT_DIR = os.path.join("out_symbolic", tag)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(path): continue
        
    df = pd.read_csv(path)
    X = df[INPUTS].values
    tables = []

    if "fexpandido" in df.columns:
        print("  > Fitting fexpandido...")
        y = df["fexpandido"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = make_model()
        model.fit(X_tr, y_tr, variable_names=["Re", "Dpin", "A", "geom"])
        rows = extract_metrics(model, X_te, y_te)
        tables.append(("fexpandido", rows))

    if "festrangulado" in df.columns:
        print("  > Fitting festrangulado...")
        y = df["festrangulado"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = make_model()
        model.fit(X_tr, y_tr, variable_names=["Re", "Dpin", "A", "geom"])
        rows = extract_metrics(model, X_te, y_te)
        tables.append(("festrangulado", rows))

    if tables:
        tex_path = os.path.join(OUT_DIR, f"results_{tag}.tex")
        with open(tex_path, "w") as f: f.write(build_latex_doc(tables))
        compile_pdf(tex_path, OUT_DIR)

print("\nDone.")