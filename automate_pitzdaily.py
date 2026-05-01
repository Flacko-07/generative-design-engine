#!/usr/bin/env python3
"""
Phase 1 Automation Script – FINAL WORKING VERSION (vector parsing fixed)
"""

from pathlib import Path
import shutil
import re
import numpy as np
from scipy.spatial import KDTree
from foamlib import FoamCase

# ------------------------------------------------------------
# Robust OpenFOAM field reader (handles nested parentheses)
# ------------------------------------------------------------
def extract_outer_parens(text):
    """Find the first '(' and its matching ')' at the same nesting level."""
    start = text.find('(')
    if start == -1:
        return None
    count = 0
    for i in range(start, len(text)):
        if text[i] == '(':
            count += 1
        elif text[i] == ')':
            count -= 1
            if count == 0:
                return text[start+1:i]
    return None

def read_openfoam_field(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    # Remove C++ comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'//.*', '', content)

    # Find class
    class_match = re.search(r'class\s+(\w+);', content)
    if not class_match:
        raise ValueError(f"No 'class' found in {file_path}")
    class_name = class_match.group(1)
    is_vector = 'vector' in class_name.lower()

    # Extract internalField
    internal_match = re.search(r'internalField\s+(\w+)\s*(.*?);', content, re.DOTALL)
    if not internal_match:
        raise ValueError(f"No internalField in {file_path}")
    field_type = internal_match.group(1)
    raw_data = internal_match.group(2).strip()

    if field_type == 'uniform':
        if is_vector:
            values = np.array([float(x) for x in raw_data.strip('()').split()])
            return 'uniform_vector', values
        else:
            return 'uniform_scalar', float(raw_data)

    elif field_type == 'nonuniform':
        # Parse: List<type> N ( ... )
        list_match = re.match(r'List<(scalar|vector)>\s*(\d+)', raw_data)
        if not list_match:
            raise ValueError(f"Unexpected nonuniform format in {file_path}")
        data_type = list_match.group(1)
        n_cells = int(list_match.group(2))

        rest = raw_data[list_match.end():].strip()
        if rest.startswith('('):
            data_str = extract_outer_parens(rest)
        else:
            paren_pos = rest.find('(')
            if paren_pos == -1:
                raise ValueError(f"Cannot find opening parenthesis in {file_path}")
            data_str = extract_outer_parens(rest[paren_pos:])

        if data_str is None:
            raise ValueError(f"Failed to extract data block from {file_path}")

        if data_type == 'scalar':
            values = np.fromstring(data_str, sep=' ')
            return 'scalar', values
        else:  # vector
            vectors = re.findall(r'\(([^)]+)\)', data_str)
            values = np.array([list(map(float, v.split())) for v in vectors])
            return 'vector', values
    else:
        raise ValueError(f"Unknown internalField type: {field_type}")

# ------------------------------------------------------------
# Main automation
# ------------------------------------------------------------
def main():
    case_dir = Path.cwd() / "automated_pitzDaily"
    if case_dir.exists():
        print(f"🧹 Removing existing case at: {case_dir}")
        shutil.rmtree(case_dir)

    print(f"📁 Creating new case at: {case_dir}")
    case = FoamCase(case_dir)
    print("✅ Case directory created")
    system_dir = case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    constant_dir = case_dir / "constant"
    constant_dir.mkdir(parents=True, exist_ok=True)

    # --- blockMeshDict ---
    with case.block_mesh_dict as f:
        f["scale"] = 1.0
        f["vertices"] = [
            [0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],
            [0, 0, 0.1], [2, 0, 0.1], [2, 1, 0.1], [0, 1, 0.1]
        ]
        f["blocks"] = [
            "hex",
            [0, 1, 2, 3, 4, 5, 6, 7],
            [100, 50, 1],
            "simpleGrading",
            [1, 1, 1]
        ]
        f["edges"] = []
        f["boundary"] = [
            ("inlet", {"type": "patch", "faces": [[0, 4, 7, 3]]}),
            ("outlet", {"type": "patch", "faces": [[1, 2, 6, 5]]}),
            ("bottom", {"type": "wall", "faces": [[0, 1, 5, 4]]}),
            ("top", {"type": "wall", "faces": [[3, 7, 6, 2]]}),
            ("frontAndBack", {"type": "empty", "faces": [[0, 3, 2, 1], [4, 5, 6, 7]]})
        ]
        f["mergePatchPairs"] = []
    print("✅ Written blockMeshDict")

    # --- controlDict ---
    with case.control_dict as f:
        f["application"] = "simpleFoam"
        f["startFrom"] = "startTime"
        f["startTime"] = 0
        f["stopAt"] = "endTime"
        f["endTime"] = 500
        f["deltaT"] = 1
        f["writeInterval"] = 100
    print("✅ Written controlDict")

    # --- fvSchemes ---
    fv_schemes_dict = {
        "ddtSchemes": {"default": "steadyState"},
        "gradSchemes": {"default": "Gauss linear"},
        "divSchemes": {"default": "none", "div(phi,U)": "bounded Gauss linearUpwind grad(U)"},
        "laplacianSchemes": {"default": "Gauss linear corrected"},
        "interpolationSchemes": {"default": "linear"},
        "snGradSchemes": {"default": "corrected"}
    }
    write_foam_dict(system_dir / "fvSchemes", "dictionary", fv_schemes_dict)
    print("✅ Written fvSchemes")

    # --- fvSolution ---
    fv_solution_dict = {
        "solvers": {
            "p": {"solver": "PCG", "preconditioner": "DIC", "tolerance": 1e-6, "relTol": 0.01},
            "U": {"solver": "smoothSolver", "smoother": "symGaussSeidel", "tolerance": 1e-5, "relTol": 0.1}
        },
        "SIMPLE": {"nNonOrthogonalCorrectors": 0, "consistent": "yes"},
        "relaxationFactors": {"equations": {"U": 0.7, "p": 0.3}}
    }
    write_foam_dict(system_dir / "fvSolution", "dictionary", fv_solution_dict)
    print("✅ Written fvSolution")

    # --- transportProperties ---
    transport_dict = {"transportModel": "Newtonian", "nu": 1e-5}
    write_foam_dict(constant_dir / "transportProperties", "dictionary", transport_dict)
    print("✅ Written transportProperties")

    # --- momentumTransport ---
    momentum_dict = {"simulationType": "laminar"}
    write_foam_dict(constant_dir / "momentumTransport", "dictionary", momentum_dict)
    print("✅ Written momentumTransport (laminar)")

    # --- 0/U and 0/p ---
    zero_dir = case_dir / "0"
    zero_dir.mkdir(parents=True, exist_ok=True)
    write_foam_field(zero_dir / "U", "volVectorField",
                     {'dimensions': '[0 1 -1 0 0 0 0]', 'type': 'uniform', 'value': [0, 0, 0]},
                     {"inlet": {"type": "fixedValue", "value": [1.0, 0.0, 0.0]},
                      "outlet": {"type": "zeroGradient"},
                      "top": {"type": "noSlip"},
                      "bottom": {"type": "noSlip"},
                      "frontAndBack": {"type": "empty"}})
    print("✅ Written 0/U")
    write_foam_field(zero_dir / "p", "volScalarField",
                     {'dimensions': '[0 2 -2 0 0 0 0]', 'type': 'uniform', 'value': 0.0},
                     {"inlet": {"type": "zeroGradient"},
                      "outlet": {"type": "fixedValue", "value": "uniform 0.0"},
                      "top": {"type": "zeroGradient"},
                      "bottom": {"type": "zeroGradient"},
                      "frontAndBack": {"type": "empty"}})
    print("✅ Written 0/p")

    # --- Run simulation ---
    print("🏃 Running blockMesh...")
    case.run("blockMesh")
    print("🏃 Running simpleFoam...")
    case.run("simpleFoam")
    print("✅ Simulation complete!")

    # --- Extract centerline pressure ---
    print("📊 Extracting centerline pressure...")
    time_dirs = [d for d in case_dir.iterdir()
                 if d.is_dir() and re.match(r'^\d+(\.\d+)?$', d.name)]
    if not time_dirs:
        raise RuntimeError("No time directories found")
    latest_time = sorted(time_dirs, key=lambda d: float(d.name))[-1].name

    case.run("postProcess -func writeCellCentres -latestTime")

    cc_path = case_dir / latest_time / "C"
    p_path = case_dir / latest_time / "p"

    cc_type, cc_data = read_openfoam_field(cc_path)
    p_type, p_data = read_openfoam_field(p_path)

    print(f"cc_type = {cc_type}, cc_data shape = {np.array(cc_data).shape}")
    print(f"p_type = {p_type}, p_data shape = {np.array(p_data).shape}")

    # Handle uniform vector if needed
    if cc_type == 'uniform_vector':
        if p_type != 'scalar':
            raise RuntimeError("Unexpected p field type")
        n_cells = len(p_data)
        cc_data = np.tile(cc_data, (n_cells, 1))
    elif cc_type == 'vector':
        if cc_data.ndim == 1:
            cc_data = cc_data.reshape(-1, 3)
    else:
        raise RuntimeError(f"Unexpected cc_type: {cc_type}")

    internal_cc = cc_data

    # Final shape check
    if internal_cc.ndim == 1:
        if internal_cc.size % 3 == 0:
            internal_cc = internal_cc.reshape(-1, 3)
        else:
            raise ValueError(f"internal_cc is 1D with size {internal_cc.size}, not a multiple of 3")

    if p_type == 'scalar':
        internal_p = np.array(p_data).flatten()
    elif p_type == 'uniform_scalar':
        internal_p = np.full(internal_cc.shape[0], p_data)
    else:
        raise RuntimeError(f"Unexpected p_type: {p_type}")

    print(f"Final internal_cc shape = {internal_cc.shape}")
    print(f"Final internal_p shape = {internal_p.shape}")

    n_points = 100
    x_vals = np.linspace(0.0, 2.0, n_points)
    y_val, z_val = 0.5, 0.05
    line_points = np.column_stack([x_vals, np.full(n_points, y_val), np.full(n_points, z_val)])

    tree = KDTree(internal_cc)
    _, indices = tree.query(line_points)
    p_line = internal_p[indices]

    output_dir = case_dir / "postProcessing"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "centerline_p.csv",
               np.column_stack([x_vals, p_line]),
               delimiter=",", header="x, p", comments="")
    print(f"✅ Centerline pressure saved to {output_dir / 'centerline_p.csv'}")

# ------------------------------------------------------------
# Helper functions (unchanged)
# ------------------------------------------------------------
def openfoam_value(val):
    if isinstance(val, list):
        return f"({' '.join(map(str, val))})"
    elif isinstance(val, tuple):
        return f"({' '.join(map(str, val))})"
    else:
        return str(val)

def write_foam_field(file_path, class_name, internal_field, boundary_field):
    content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    location    "0";
    object      {file_path.stem};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      {internal_field['dimensions']};

internalField   {internal_field['type']} {openfoam_value(internal_field['value'])};

boundaryField
{{
"""
    for patch, spec in boundary_field.items():
        content += f"    {patch}\n    {{\n"
        for key, val in spec.items():
            if key == "value":
                if isinstance(val, list):
                    content += f"        value       uniform {openfoam_value(val)};\n"
                else:
                    content += f"        value       {val};\n"
            else:
                content += f"        {key:<14} {val};\n"
        content += "    }\n"
    content += "}\n\n// ************************************************************************* //"
    with open(file_path, 'w') as f:
        f.write(content)

def write_foam_dict(file_path, class_name, content_dict):
    header = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    location    "{file_path.parent.name}";
    object      {file_path.name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
    def write_dict(d, indent=0):
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{'    '*indent}{key}")
                lines.append(f"{'    '*indent}{{")
                lines.extend(write_dict(value, indent+1))
                lines.append(f"{'    '*indent}}}")
            elif isinstance(value, list):
                items = ' '.join(str(v) for v in value)
                lines.append(f"{'    '*indent}{key} ({items});")
            elif isinstance(value, str) and value.endswith(';'):
                lines.append(f"{'    '*indent}{key} {value}")
            else:
                lines.append(f"{'    '*indent}{key} {value};")
        return lines
    body = "\n".join(write_dict(content_dict))
    with open(file_path, 'w') as f:
        f.write(header + body + "\n\n// ************************************************************************* //")

if __name__ == "__main__":
    main()