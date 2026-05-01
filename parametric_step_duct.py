#!/usr/bin/env python3
"""
Phase 2 – Parametric Step Duct + Batch Runner (blockMeshDict manually written)
"""

from pathlib import Path
import shutil
import re
import numpy as np
from scipy.spatial import KDTree
from foamlib import FoamCase

# ----------------------------------------------------------------------
# Field reader & helper functions (same as Phase 1 final version)
# ----------------------------------------------------------------------
def extract_outer_parens(text):
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
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'//.*', '', content)
    class_match = re.search(r'class\s+(\w+);', content)
    if not class_match:
        raise ValueError(f"No 'class' found in {file_path}")
    class_name = class_match.group(1)
    is_vector = 'vector' in class_name.lower()
    internal_match = re.search(r'internalField\s+(\w+)\s*(.*?);', content, re.DOTALL)
    if not internal_match:
        raise ValueError(f"No internalField in {file_path}")
    field_type = internal_match.group(1)
    raw_data = internal_match.group(2).strip()
    if field_type == 'uniform':
        if is_vector:
            vals = np.array([float(x) for x in raw_data.strip('()').split()])
            return 'uniform_vector', vals
        else:
            return 'uniform_scalar', float(raw_data)
    elif field_type == 'nonuniform':
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
            return 'scalar', np.fromstring(data_str, sep=' ')
        else:  # vector
            vectors = re.findall(r'\(([^)]+)\)', data_str)
            return 'vector', np.array([list(map(float, v.split())) for v in vectors])
    else:
        raise ValueError(f"Unknown internalField type: {field_type}")

def openfoam_value(val):
    if isinstance(val, list):
        return f"({' '.join(map(str, val))})"
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


# ----------------------------------------------------------------------
# Build a case for given step_height and step_length
# ----------------------------------------------------------------------
def build_and_run_step_case(step_height, step_length,
                            base_dir="parametric_cases",
                            total_length=5.0, channel_height=1.0,
                            nu=1e-5, U_inlet=1.0):
    """Create, mesh, run, and return reattachment length."""
    case_dir = Path(base_dir) / f"step_h{step_height}_l{step_length}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case = FoamCase(case_dir)

    system_dir = case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    constant_dir = case_dir / "constant"
    constant_dir.mkdir(parents=True, exist_ok=True)

    # Geometry definition
    thick = 0.1   # extrusion thickness (2D)
    L = total_length
    H = channel_height
    sh = step_height
    sl = step_length

    # --- build blockMeshDict as a formatted string (3 blocks) ---
    import textwrap
    block_mesh_str = textwrap.dedent(f"""\
    /*--------------------------------*- C++ -*----------------------------------*\\
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
        class       dictionary;
        location    "system";
        object      blockMeshDict;
    }}
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    convertToMeters 1.0;

    vertices
    (
        // z = 0
        (0 {sh} 0)          //0
        ({sl} {sh} 0)       //1
        ({sl} {H} 0)        //2
        (0 {H} 0)           //3
        ({sl} 0 0)          //4
        ({L} 0 0)           //5
        ({L} {H} 0)         //6
        ({L} {sh} 0)        //7
        // z = {thick}
        (0 {sh} {thick})    //8
        ({sl} {sh} {thick}) //9
        ({sl} {H} {thick})  //10
        (0 {H} {thick})     //11
        ({sl} 0 {thick})    //12
        ({L} 0 {thick})     //13
        ({L} {H} {thick})   //14
        ({L} {sh} {thick})  //15
    );

    blocks
    (
        hex (0 1 2 3 8 9 10 11) (20 20 1) simpleGrading (1 1 1)
        hex (4 5 7 1 12 13 15 9) (40 15 1) simpleGrading (1 0.5 1)
        hex (1 7 6 2 9 15 14 10) (40 15 1) simpleGrading (1 2 1)
    );

    edges
    (
    );

    boundary
    (
        inlet
        {{
            type patch;
            faces
            (
                (0 8 11 3)
            );
        }}
        outlet
        {{
            type patch;
            faces
            (
                (5 13 15 7)
                (7 15 14 6)
            );
        }}
        topWall
        {{
            type wall;
            faces
            (
                (3 2 10 11)
                (2 6 14 10)
            );
        }}
        bottomWall
        {{
            type wall;
            faces
            (
                (4 5 13 12)
            );
        }}
        stepTop
        {{
            type wall;
            faces
            (
                (1 0 8 9)
            );
        }}
        stepFace
        {{
            type wall;
            faces
            (
                (4 1 9 12)
            );
        }}
        block0_to_block1_upper
        {{
            type patch;
            faces
            (
                (1 2 10 9)
            );
        }}
        block1_upper_to_block0
        {{
            type patch;
            faces
            (
                (1 9 10 2)
            );
        }}
        frontAndBack
        {{
            type empty;
            faces
            (
                (0 3 2 1)
                (4 2 6 5)
                (4 1 2 5)    // lower block z-min? Actually lower block bottom face is (4 5 7 1)
                (8 9 10 11)
                (12 13 15 9)
                (9 15 14 10)
            );
        }}
    );

    mergePatchPairs
    (
        (block0_to_block1_upper block1_upper_to_block0)
    );

    // ************************************************************************* //
    """)
    with open(system_dir / "blockMeshDict", "w") as f:
        f.write(block_mesh_str)

    # controlDict, fvSchemes, fvSolution
    with case.control_dict as c:
        c["application"] = "simpleFoam"
        c["startFrom"] = "startTime"
        c["startTime"] = 0
        c["stopAt"] = "endTime"
        c["endTime"] = 500
        c["deltaT"] = 1
        c["writeInterval"] = 100
    write_foam_dict(system_dir / "fvSchemes", "dictionary", {
        "ddtSchemes": {"default": "steadyState"},
        "gradSchemes": {"default": "Gauss linear"},
        "divSchemes": {"default": "none", "div(phi,U)": "bounded Gauss linearUpwind grad(U)"},
        "laplacianSchemes": {"default": "Gauss linear corrected"},
        "interpolationSchemes": {"default": "linear"},
        "snGradSchemes": {"default": "corrected"}
    })
    write_foam_dict(system_dir / "fvSolution", "dictionary", {
        "solvers": {
            "p": {"solver": "PCG", "preconditioner": "DIC", "tolerance": 1e-6, "relTol": 0.01},
            "U": {"solver": "smoothSolver", "smoother": "symGaussSeidel", "tolerance": 1e-5, "relTol": 0.1}
        },
        "SIMPLE": {"nNonOrthogonalCorrectors": 0, "consistent": "yes"},
        "relaxationFactors": {"equations": {"U": 0.7, "p": 0.3}}
    })
    write_foam_dict(constant_dir / "transportProperties", "dictionary", {
        "transportModel": "Newtonian", "nu": nu
    })
    write_foam_dict(constant_dir / "momentumTransport", "dictionary", {
        "simulationType": "laminar"
    })

    # 0/U and 0/p
    zero_dir = case_dir / "0"
    zero_dir.mkdir(parents=True, exist_ok=True)
    write_foam_field(zero_dir / "U", "volVectorField",
                     {'dimensions': '[0 1 -1 0 0 0 0]', 'type': 'uniform', 'value': [U_inlet, 0, 0]},
                     {"inlet": {"type": "fixedValue", "value": [U_inlet, 0.0, 0.0]},
                      "outlet": {"type": "zeroGradient"},
                      "topWall": {"type": "noSlip"},
                      "bottomWall": {"type": "noSlip"},
                      "stepTop": {"type": "noSlip"},
                      "stepFace": {"type": "noSlip"},
                      "block0_to_block1": {"type": "empty"},
                      "block1_to_block0": {"type": "empty"},
                      "frontAndBack": {"type": "empty"}})
    write_foam_field(zero_dir / "p", "volScalarField",
                     {'dimensions': '[0 2 -2 0 0 0 0]', 'type': 'uniform', 'value': 0.0},
                     {"inlet": {"type": "zeroGradient"},
                      "outlet": {"type": "fixedValue", "value": "uniform 0.0"},
                      "topWall": {"type": "zeroGradient"},
                      "bottomWall": {"type": "zeroGradient"},
                      "stepTop": {"type": "zeroGradient"},
                      "stepFace": {"type": "zeroGradient"},
                      "block0_to_block1": {"type": "empty"},
                      "block1_to_block0": {"type": "empty"},
                      "frontAndBack": {"type": "empty"}})

    # Run
    case.run("blockMesh")
    case.run("simpleFoam")

    # Extract reattachment length (same as before)
    latest_time = sorted([d for d in case_dir.iterdir()
                          if d.is_dir() and re.match(r'^\d+(\.\d+)?$', d.name)],
                         key=lambda d: float(d.name))[-1].name
    case.run("postProcess -func writeCellCentres -latestTime")
    U_path = case_dir / latest_time / "U"
    C_path = case_dir / latest_time / "C"
    U_type, U_data = read_openfoam_field(U_path)
    C_type, C_data = read_openfoam_field(C_path)
    if U_data.shape[0] != C_data.shape[0]:
        raise RuntimeError("Mismatch between U and C cell counts")
    u_x = U_data[:, 0]
    x = C_data[:, 0]
    downstream = x > step_length
    x_ds = x[downstream]
    u_ds = u_x[downstream]
    sort_idx = np.argsort(x_ds)
    x_ds = x_ds[sort_idx]
    u_ds = u_ds[sort_idx]
    reattach_x = None
    for i in range(len(u_ds)-1):
        if u_ds[i] < 0 and u_ds[i+1] >= 0:
            x0, x1 = x_ds[i], x_ds[i+1]
            u0, u1 = u_ds[i], u_ds[i+1]
            reattach_x = x0 - u0 * (x1 - x0) / (u1 - u0)
            break
    if reattach_x is None:
        reattach_x = x_ds[-1]
    return reattach_x


# ----------------------------------------------------------------------
# Batch Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    step_heights = np.linspace(0.1, 0.4, 4)
    step_lengths = np.linspace(0.5, 2.0, 4)

    results = []
    for sh in step_heights:
        for sl in step_lengths:
            print(f"Running step h={sh:.2f}, L={sl:.2f} ...", end=" ")
            reattach = build_and_run_step_case(sh, sl)
            results.append((sh, sl, reattach))
            print(f"reattachment x = {reattach:.3f}")

    import csv
    with open("step_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step_height", "step_length", "reattachment_x"])
        writer.writerows(results)
    print("✅ Dataset saved to step_dataset.csv")