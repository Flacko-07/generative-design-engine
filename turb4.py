#!/usr/bin/env python3
"""Generate 500 turbulent 4‑parameter cylinder cases (d, x, U, channel_height)."""

import csv, re, shutil, textwrap, time
from pathlib import Path
import numpy as np
from foamlib import FoamCase
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ═══════════════════════════════════════════════════════════
# Helper functions (identical to previous working scripts)
# ═══════════════════════════════════════════════════════════
def openfoam_value(val):
    if isinstance(val, (list, tuple)):
        return f"({' '.join(map(str, val))})"
    return str(val)

def write_foam_field(file_path, class_name, internal_field, boundary_field):
    content = f"""\
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
        for k, v in spec.items():
            if k == "value":
                if isinstance(v, list):
                    content += f"        value       uniform {openfoam_value(v)};\n"
                else:
                    content += f"        value       {v};\n"
            else:
                content += f"        {k:<14} {v};\n"
        content += "    }\n"
    content += "}\n\n// ************************************************************************* //"
    with open(file_path, 'w') as f:
        f.write(content)

def write_foam_dict(file_path, class_name, content_dict):
    header = f"""\
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
    class       {class_name};
    location    "{file_path.parent.name}";
    object      {file_path.name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
    def write_dict(d, indent=0):
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{'    ' * indent}{k}")
                lines.append(f"{'    ' * indent}{{")
                lines.extend(write_dict(v, indent + 1))
                lines.append(f"{'    ' * indent}}}")
            elif isinstance(v, list):
                items = ' '.join(str(x) for x in v)
                lines.append(f"{'    ' * indent}{k} ({items});")
            elif isinstance(v, str) and v.endswith(';'):
                lines.append(f"{'    ' * indent}{k} {v}")
            else:
                lines.append(f"{'    ' * indent}{k} {v};")
        return lines
    body = "\n".join(write_dict(content_dict))
    with open(file_path, 'w') as f:
        f.write(header + body + "\n\n// ************************************************************************* //")

def _write_facet(f, normal, v0, v1, v2):
    f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
    f.write("    outer loop\n")
    for v in (v0, v1, v2):
        f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    f.write("    endloop\n  endfacet\n")

def write_cylinder_stl(path, diameter, x_center, y_center, z_min, z_max, num_faces=36):
    r = diameter / 2.0
    angles = np.linspace(0, 2 * np.pi, num_faces + 1)[:-1]
    bottom = np.array([(x_center + r * np.cos(th),
                        y_center + r * np.sin(th), z_min) for th in angles])
    top    = np.array([(x_center + r * np.cos(th),
                        y_center + r * np.sin(th), z_max) for th in angles])
    bc = np.array([x_center, y_center, z_min])
    tc = np.array([x_center, y_center, z_max])

    with open(path, 'w') as f:
        f.write("solid cylinder\n")
        for i in range(num_faces):
            nxt = (i + 1) % num_faces
            _write_facet(f, (0, 0, -1), bc, bottom[nxt], bottom[i])
        for i in range(num_faces):
            nxt = (i + 1) % num_faces
            _write_facet(f, (0, 0, 1), tc, top[i], top[nxt])
        for i in range(num_faces):
            nxt = (i + 1) % num_faces
            mid_angle = (angles[i] + angles[nxt]) / 2.0
            nx, ny = np.cos(mid_angle), np.sin(mid_angle)
            _write_facet(f, (nx, ny, 0), bottom[i], bottom[nxt], top[i])
            _write_facet(f, (nx, ny, 0), top[i],    bottom[nxt], top[nxt])
        f.write("endsolid cylinder\n")

def find_forces_file(case_dir):
    pp_dir = case_dir / "postProcessing"
    for name in ("forces.dat", "force.dat"):
        candidates = sorted(pp_dir.rglob(name))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"No forces file found in {pp_dir}")

def parse_forces_file(forces_file):
    with open(forces_file) as fh:
        data_lines = [ln for ln in fh if ln.strip() and not ln.startswith('#')]
    last = data_lines[-1]
    tokens = last.replace('(', ' ').replace(')', ' ').split()
    fx, fy, fz = float(tokens[1]), float(tokens[2]), float(tokens[3])
    return fx, fy, fz

# ═══════════════════════════════════════════════════════════
# Turbulent case builder with variable channel height
# ═══════════════════════════════════════════════════════════
def build_turbulent_case_4param(diameter, x_center, U_inlet, channel_height,
                                nu=1.5e-5, y_center=0.0,
                                base_dir="turbulent_4param_cases",
                                channel_length=5.0, channel_width=0.5,
                                target_cell_size=0.05):
    Re = U_inlet * diameter / nu

    case_dir = Path(base_dir) / f"turb4_D{diameter:.3f}_X{x_center:.3f}_U{U_inlet:.1f}_H{channel_height:.3f}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case = FoamCase(case_dir)

    system_dir   = case_dir / "system";   system_dir.mkdir(parents=True, exist_ok=True)
    constant_dir = case_dir / "constant"; constant_dir.mkdir(parents=True, exist_ok=True)
    tri_dir      = constant_dir / "triSurface"; tri_dir.mkdir(parents=True, exist_ok=True)

    write_cylinder_stl(tri_dir / "cylinder.stl", diameter, x_center, y_center,
                       0.0, channel_width, num_faces=36)

    H = channel_height        # <-- this is the new independent variable
    W = channel_width
    L = channel_length
    nx = max(10, int(round(L / target_cell_size)))
    ny = max(5,  int(round(H / target_cell_size)))
    nz = max(1,  int(round(W / target_cell_size)))

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
        (0 {-H/2} 0)
        ({L} {-H/2} 0)
        ({L} {H/2} 0)
        (0 {H/2} 0)
        (0 {-H/2} {W})
        ({L} {-H/2} {W})
        ({L} {H/2} {W})
        (0 {H/2} {W})
    );

    blocks
    (
        hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
    );

    edges
    (
    );

    boundary
    (
        inlet
        {{
            type patch;
            faces ((0 4 7 3));
        }}
        outlet
        {{
            type patch;
            faces ((1 2 6 5));
        }}
        top
        {{
            type wall;
            faces ((3 7 6 2));
        }}
        bottom
        {{
            type wall;
            faces ((0 1 5 4));
        }}
        frontAndBack
        {{
            type symmetry;
            faces ((0 3 2 1) (4 5 6 7));
        }}
    );

    // ************************************************************************* //
    """)
    with open(system_dir / "blockMeshDict", "w") as f:
        f.write(block_mesh_str)

    snappy_str = textwrap.dedent(f"""\
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
        object      snappyHexMeshDict;
    }}
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    castellatedMesh true;
    snap            true;
    addLayers       false;
    mergeTolerance  1e-6;

    geometry
    {{
        cylinder
        {{
            type triSurfaceMesh;
            file "cylinder.stl";
        }}
    }}

    castellatedMeshControls
    {{
        features ();

        refinementSurfaces
        {{
            cylinder
            {{
                level (3 3);
            }}
        }}

        refinementRegions
        {{
        }}

        resolveFeatureAngle  30;
        locationInMesh (0.1 0.0 {W / 2});
        maxLocalCells        1000000;
        maxGlobalCells       2000000;
        minRefinementCells   10;
        nCellsBetweenLevels  3;
        allowFreeStandingZoneFaces true;
    }}

    snapControls
    {{
        nSmoothPatch 3;
        tolerance    2.0;
        nSolveIter   30;
        nRelaxIter   5;
    }}

    meshQualityControls
    {{
        maxNonOrtho          65;
        maxBoundarySkewness  20;
        maxInternalSkewness  4;
        maxConcave           80;
        minVol               1e-13;
        minTetQuality        1e-30;
        minArea              -1;
        minTwist             0.05;
        minDeterminant       0.001;
        minFaceWeight        0.05;
        minVolRatio          0.01;
        minTriangleTwist     -1;
        nSmoothScale         4;
        errorReduction       0.75;
    }}

    addLayersControls
    {{
        relativeSizes false;
    }}

    // ************************************************************************* //
    """)
    with open(system_dir / "snappyHexMeshDict", "w") as f:
        f.write(snappy_str)

    _cd = [
        "/*--------------------------------*- C++ -*----------------------------------*\\",
        "| =========                 |                                                 |",
        "| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |",
        "|  \\    /   O peration     | Version:  v2406                                 |",
        "|   \\  /    A nd           | Website:  www.openfoam.com                      |",
        "|    \\/     M anipulation  |                                                 |",
        "\\*---------------------------------------------------------------------------*/",
        "FoamFile", "{",
        '    version     2.0;', '    format      ascii;',
        '    class       dictionary;', '    location    "system";',
        '    object      controlDict;', "}",
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //",
        "", "application     simpleFoam;", "startFrom       startTime;",
        "startTime       0;", "stopAt          endTime;", "endTime         500;",
        "deltaT          1;", "writeControl    timeStep;", "writeInterval   100;",
        "purgeWrite      0;", "writeFormat     ascii;", "writePrecision  6;",
        "runTimeModifiable true;", "",
        "functions", "{",
        "    forces", "    {",
        "        type            forces;",
        "        libs            (\"libforces.so\");",
        "        patches         (cylinder);",
        "        rho             rhoInf;",
        "        rhoInf          1.0;",
        "        CofR            (0 0 0);",
        "        log             true;",
        "        writeControl    timeStep;",
        "        writeInterval   1;",
        "    }", "}",
        "", "// ************************************************************************* //",
    ]
    with open(system_dir / "controlDict", "w") as f:
        f.write("\n".join(_cd) + "\n")

    write_foam_dict(system_dir / "fvSchemes", "dictionary", {
        "ddtSchemes":          {"default": "steadyState"},
        "gradSchemes":         {"default": "Gauss linear"},
        "divSchemes":          {
            "default":         "none",
            "div(phi,U)":      "bounded Gauss linearUpwind grad(U)",
            "div(phi,k)":      "bounded Gauss upwind",
            "div(phi,omega)":  "bounded Gauss upwind"
        },
        "laplacianSchemes":    {"default": "Gauss linear corrected"},
        "interpolationSchemes":{"default": "linear"},
        "snGradSchemes":       {"default": "corrected"},
        "wallDist":            {"method": "meshWave"}
    })
    write_foam_dict(system_dir / "fvSolution", "dictionary", {
        "solvers": {
            "p": {"solver": "PCG",          "preconditioner": "DIC",
                  "tolerance": 1e-6,        "relTol": 0.01},
            "U": {"solver": "smoothSolver", "smoother": "symGaussSeidel",
                  "tolerance": 1e-5,        "relTol": 0.1},
            "k": {"solver": "smoothSolver", "smoother": "symGaussSeidel",
                  "tolerance": 1e-5,        "relTol": 0.1},
            "omega": {"solver": "smoothSolver", "smoother": "symGaussSeidel",
                      "tolerance": 1e-5,    "relTol": 0.1}
        },
        "SIMPLE": {"nNonOrthogonalCorrectors": 0, "consistent": "yes"},
        "relaxationFactors": {"equations": {"U": 0.7, "p": 0.3, "k": 0.7, "omega": 0.7}}
    })
    write_foam_dict(constant_dir / "transportProperties", "dictionary",
                    {"transportModel": "Newtonian", "nu": nu})
    write_foam_dict(constant_dir / "momentumTransport", "dictionary", {
        "simulationType": "RAS",
        "RAS": {
            "model": "kOmegaSST",
            "turbulence": "true",
            "printCoeffs": "true"
        }
    })

    zero_dir = case_dir / "0"; zero_dir.mkdir(parents=True, exist_ok=True)
    I = 0.05
    L_turb = 0.1 * diameter
    k_inlet = 1.5 * (U_inlet * I)**2
    omega_inlet = np.sqrt(k_inlet) / (0.09 * L_turb)

    write_foam_field(zero_dir / "U", "volVectorField",
                     {"dimensions": "[0 1 -1 0 0 0 0]", "type": "uniform", "value": [U_inlet, 0, 0]},
                     {
                         "inlet": {"type": "fixedValue", "value": [U_inlet, 0.0, 0.0]},
                         "outlet": {"type": "zeroGradient"},
                         "top": {"type": "slip"},
                         "bottom": {"type": "slip"},
                         "cylinder": {"type": "noSlip"},
                         "frontAndBack": {"type": "symmetry"}
                     })
    write_foam_field(zero_dir / "p", "volScalarField",
                     {"dimensions": "[0 2 -2 0 0 0 0]", "type": "uniform", "value": 0.0},
                     {
                         "inlet": {"type": "zeroGradient"},
                         "outlet": {"type": "fixedValue", "value": "uniform 0.0"},
                         "top": {"type": "zeroGradient"},
                         "bottom": {"type": "zeroGradient"},
                         "cylinder": {"type": "zeroGradient"},
                         "frontAndBack": {"type": "symmetry"}
                     })
    write_foam_field(zero_dir / "k", "volScalarField",
                     {"dimensions": "[0 2 -2 0 0 0 0]", "type": "uniform", "value": k_inlet},
                     {
                         "inlet": {"type": "fixedValue", "value": f"uniform {k_inlet}"},
                         "outlet": {"type": "zeroGradient"},
                         "top": {"type": "zeroGradient"},
                         "bottom": {"type": "zeroGradient"},
                         "cylinder": {"type": "kqRWallFunction", "value": "uniform 1e-10"},
                         "frontAndBack": {"type": "symmetry"}
                     })
    write_foam_field(zero_dir / "omega", "volScalarField",
                     {"dimensions": "[0 0 -1 0 0 0 0]", "type": "uniform", "value": omega_inlet},
                     {
                         "inlet": {"type": "fixedValue", "value": f"uniform {omega_inlet}"},
                         "outlet": {"type": "zeroGradient"},
                         "top": {"type": "zeroGradient"},
                         "bottom": {"type": "zeroGradient"},
                         "cylinder": {"type": "omegaWallFunction", "value": "uniform 1000"},
                         "frontAndBack": {"type": "symmetry"}
                     })
    write_foam_field(zero_dir / "nut", "volScalarField",
                     {"dimensions": "[0 2 -1 0 0 0 0]", "type": "uniform", "value": 0.0},
                     {
                         "inlet":        {"type": "calculated", "value": "uniform 0"},
                         "outlet":       {"type": "calculated", "value": "uniform 0"},
                         "top":          {"type": "calculated", "value": "uniform 0"},
                         "bottom":       {"type": "calculated", "value": "uniform 0"},
                         "cylinder":     {"type": "nutkWallFunction", "value": "uniform 0"},
                         "frontAndBack": {"type": "symmetry"}
                     })

    case.run("blockMesh")
    case.run("snappyHexMesh -overwrite")

    # Patch detection & forces update (same as before)
    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
    cylinder_patch = "cylinder"
    if boundary_file.exists():
        with open(boundary_file) as bf:
            btext = bf.read()
        patch_names = re.findall(r'^\s{4}(\w+)\s*$', btext, re.MULTILINE)
        std = {'inlet', 'outlet', 'top', 'bottom', 'frontAndBack', 'defaultFaces'}
        cyl_candidates = [p for p in patch_names if p.lower() not in std and 'cyl' in p.lower()]
        if not cyl_candidates:
            cyl_candidates = [p for p in patch_names if p.lower() not in std]
        cylinder_patch = cyl_candidates[-1] if cyl_candidates else 'cylinder'
        cd_path = system_dir / "controlDict"
        cd_text = cd_path.read_text()
        cd_text = re.sub(r'(patches\s+\()([^)]+)(\);)', rf'\g<1>{cylinder_patch}\g<3>', cd_text)
        cd_path.write_text(cd_text)

    case.run("simpleFoam")

    forces_file = find_forces_file(case_dir)
    fx, _, _ = parse_forces_file(forces_file)
    A_ref = diameter * channel_width
    Cd = 2.0 * fx / (1.0 * U_inlet**2 * A_ref)
    return Re, Cd, channel_height


# ═══════════════════════════════════════════════════════════
# Worker for parallel execution
# ═══════════════════════════════════════════════════════════
def worker(args):
    d, x, U, ch = args
    Re, Cd, ch = build_turbulent_case_4param(d, x, U, ch)
    return d, x, U, ch, Re, Cd

# ═══════════════════════════════════════════════════════════
# Main batch generation
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    sampler = qmc.LatinHypercube(d=4, seed=42)
    sample = sampler.random(n=500)
    l_bounds = np.array([0.05, 1.0, 5.0, 1.5])
    u_bounds = np.array([0.50, 3.0, 20.0, 4.0])
    scaled = sample * (u_bounds - l_bounds) + l_bounds

    total = len(scaled)
    print(f"🚀 Launching {total} turbulent 4‑parameter cases...")
    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(worker, row): row for row in scaled}
        for fut in as_completed(futures):
            try:
                d, x, U, ch, Re, Cd = fut.result()
                results.append((d, x, U, ch, Re, Cd))
                print(f"  D={d:.3f} x={x:.3f} U={U:.1f} ch={ch:.3f} Re={Re:.0f} Cd={Cd:.4f}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")

    with open("turbulent_4param.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["diameter", "x_center", "U_inlet", "channel_height", "Re", "Cd"])
        writer.writerows(results)
    print(f"\n✅ Saved {len(results)} cases to turbulent_4param.csv ({time.time()-start_time:.0f}s)")