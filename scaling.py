#!/usr/bin/env python3
"""
Phase 2.5 – SCALED Parametric Cylinder (400 designs, parallel)
Save this as parametric_cylinder_scaled.py and run.
"""

import csv, re, shutil, textwrap, time, random
from pathlib import Path
import numpy as np
from foamlib import FoamCase
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ══════════════════════════════════════════════════════════════════════════════
#  INSERT HERE ALL THE HELPER FUNCTIONS FROM YOUR WORKING SCRIPT:
#  openfoam_value, write_foam_field, write_foam_dict, _write_facet,
#  write_cylinder_stl, find_forces_file, parse_forces_file,
#  AND the main build_and_run_cylinder function.
#
#  (I cannot paste them because they are identical to your last working
#   parametric_cylinder.py – just copy them right below this line.)
# ══════════════════════════════════════════════════════════════════════════════
def openfoam_value(val):
    """Format a Python value as an OpenFOAM literal."""
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


# ──────────────────────────────────────────────────────────────────────────────
# STL generation – FIX: proper outward normals on side facets
# ──────────────────────────────────────────────────────────────────────────────

def _write_facet(f, normal, v0, v1, v2):
    f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
    f.write("    outer loop\n")
    for v in (v0, v1, v2):
        f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    f.write("    endloop\n  endfacet\n")


def write_cylinder_stl(path, diameter, x_center, y_center, z_min, z_max, num_faces=36):
    """
    Write a closed cylinder STL.
    Side-facet normals are now the true outward radial unit normals,
    not (0,0,0) which is invalid per ASCII STL spec.
    """
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

        # Bottom cap  (normal points -z)
        for i in range(num_faces):
            nxt = (i + 1) % num_faces
            _write_facet(f, (0, 0, -1), bc, bottom[nxt], bottom[i])

        # Top cap  (normal points +z)
        for i in range(num_faces):
            nxt = (i + 1) % num_faces
            _write_facet(f, (0, 0, 1), tc, top[i], top[nxt])

        # Side quads – two triangles per quad, outward radial normal
        for i in range(num_faces):
            nxt = (i + 1) % num_faces
            # Outward normal = average of the two edge angles in XY, z=0
            mid_angle = (angles[i] + angles[nxt]) / 2.0
            nx = np.cos(mid_angle)
            ny = np.sin(mid_angle)
            # Triangle 1: bottom-i, bottom-nxt, top-i
            _write_facet(f, (nx, ny, 0), bottom[i], bottom[nxt], top[i])
            # Triangle 2: top-i, bottom-nxt, top-nxt
            _write_facet(f, (nx, ny, 0), top[i],    bottom[nxt], top[nxt])

        f.write("endsolid cylinder\n")


# ──────────────────────────────────────────────────────────────────────────────
# Force file parser
# ──────────────────────────────────────────────────────────────────────────────

def find_forces_file(case_dir):
    """
    Locate the forces output file written by OpenFOAM.

    OF version history:
      < OF8   : postProcessing/forces/<time>/force.dat
      OF8-12  : postProcessing/forces/<time>/force.dat  (same)
      OF13+   : postProcessing/forces/<time>/forces.dat (plural!)

    We search for BOTH names under the entire postProcessing tree
    and print a diagnostic listing so failures are easy to diagnose.
    """
    pp_dir = case_dir / "postProcessing"

    # Always print what postProcessing actually contains for diagnostics
    if pp_dir.exists():
        all_files = list(pp_dir.rglob("*"))
        print(f"\n    [diag] postProcessing tree ({len(all_files)} items):")
        for p in sorted(all_files)[:30]:   # cap at 30 lines
            print(f"           {p.relative_to(case_dir)}")
        if len(all_files) > 30:
            print(f"           ... ({len(all_files) - 30} more)")
    else:
        print("\n    [diag] postProcessing directory does not exist!")

    # Search for any forces output file (both naming conventions)
    for name in ("forces.dat", "force.dat"):
        candidates = sorted(pp_dir.rglob(name))
        if candidates:
            chosen = candidates[-1]   # latest time directory
            print(f"    [diag] Using forces file: {chosen.relative_to(case_dir)}")
            return chosen

    # Nothing found — give a helpful error with the actual tree
    raise FileNotFoundError(
        f"No forces output file (forces.dat / force.dat) found anywhere under "
        f"{pp_dir}. Check the patch name in the forces function matches the "
        f"actual snappyHexMesh patch name, and that libforces.so loaded correctly."
    )


def parse_forces_file(forces_file):
    """
    Return (Fx, Fy, Fz) from the last data line of force.dat.
    Lines beginning with '#' are comments/headers and are skipped.
    Typical format per line:
        time    (fx fy fz)    (mx my mz)   ...
    In older OF, columns are space-separated scalars; in newer OF they
    are grouped as vectors in parentheses.  We handle both.
    """
    with open(forces_file) as fh:
        data_lines = [ln for ln in fh if ln.strip() and not ln.startswith('#')]
    if not data_lines:
        raise ValueError(f"No data lines found in {forces_file}")

    last = data_lines[-1]
    # Strip parentheses so we get bare numbers regardless of OF version
    tokens = last.replace('(', ' ').replace(')', ' ').split()
    # tokens[0] = time, tokens[1..3] = Fx Fy Fz (pressure+viscous total)
    fx, fy, fz = float(tokens[1]), float(tokens[2]), float(tokens[3])
    return fx, fy, fz


# ──────────────────────────────────────────────────────────────────────────────
# Main case builder
# ──────────────────────────────────────────────────────────────────────────────

def build_and_run_cylinder(
    diameter,
    x_center,
    y_center=0.0,
    base_dir="cylinder_cases",
    channel_length=5.0,
    channel_height=2.0,
    channel_width=0.5,
    U_inlet=1.0,
    nu=1e-5,
    target_cell_size=0.1,   # FIX: drives background mesh resolution adaptively
):
    """
    Build, mesh, run, and return (Re, Cd) for one cylinder configuration.
    """
    # ── Reynolds number check ──────────────────────────────────────────────
    Re = U_inlet * diameter / nu
    print(f"    Re = {Re:.1f}", end="")
    if Re > 2000:
        print(f"  ⚠️  Re > 2000 – flow is likely turbulent; laminar solver may be inaccurate!", end="")

    case_dir = Path(base_dir) / f"cyl_D{diameter:.3f}_X{x_center:.3f}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case = FoamCase(case_dir)

    system_dir   = case_dir / "system";   system_dir.mkdir(parents=True, exist_ok=True)
    constant_dir = case_dir / "constant"; constant_dir.mkdir(parents=True, exist_ok=True)
    tri_surface_dir = constant_dir / "triSurface"; tri_surface_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate STL (with corrected normals)
    write_cylinder_stl(
        tri_surface_dir / "cylinder.stl",
        diameter, x_center, y_center,
        0.0, channel_width, num_faces=36
    )

    H = channel_height
    W = channel_width
    L = channel_length

    # 2. Background mesh  ── FIX: derive cell counts from target_cell_size
    nx = max(10, int(round(L / target_cell_size)))
    ny = max(5,  int(round(H / target_cell_size)))
    nz = max(1,  int(round(W / target_cell_size)))
    print(f"  mesh=({nx}×{ny}×{nz})", end="")

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

    # 3. snappyHexMeshDict
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
                level (2 2);
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

    # 4. controlDict – written as raw text to avoid foamlib serialization bugs.
    #    foamlib converts "true"/"false" strings to Python booleans and mangles
    #    nested dicts, producing invalid OpenFOAM syntax for the forces function.
    #    Also: OF13 uses `libs (forces);` not `libs ("libforces.so");`
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

    # 5. fvSchemes, fvSolution, transportProperties, momentumTransport
    write_foam_dict(system_dir / "fvSchemes", "dictionary", {
        "ddtSchemes":          {"default": "steadyState"},
        "gradSchemes":         {"default": "Gauss linear"},
        "divSchemes":          {"default": "none",
                                "div(phi,U)": "bounded Gauss linearUpwind grad(U)"},
        "laplacianSchemes":    {"default": "Gauss linear corrected"},
        "interpolationSchemes":{"default": "linear"},
        "snGradSchemes":       {"default": "corrected"},
    })
    write_foam_dict(system_dir / "fvSolution", "dictionary", {
        "solvers": {
            "p": {"solver": "PCG",          "preconditioner": "DIC",
                  "tolerance": 1e-6,        "relTol": 0.01},
            "U": {"solver": "smoothSolver", "smoother": "symGaussSeidel",
                  "tolerance": 1e-5,        "relTol": 0.1},
        },
        "SIMPLE":           {"nNonOrthogonalCorrectors": 0, "consistent": "yes"},
        "relaxationFactors": {"equations": {"U": 0.7, "p": 0.3}},
    })
    write_foam_dict(constant_dir / "transportProperties", "dictionary",
                    {"transportModel": "Newtonian", "nu": nu})
    write_foam_dict(constant_dir / "momentumTransport", "dictionary",
                    {"simulationType": "laminar"})

    # 6. Initial conditions
    #    FIX: top/bottom are far-field boundaries → slip, not noSlip
    #    noSlip would add a no-slip wall effect across the entire channel,
    #    artificially confining the flow and biasing Cd.
    zero_dir = case_dir / "0"
    zero_dir.mkdir(parents=True, exist_ok=True)

    write_foam_field(
        zero_dir / "U", "volVectorField",
        {"dimensions": "[0 1 -1 0 0 0 0]", "type": "uniform", "value": [U_inlet, 0, 0]},
        {
            "inlet":        {"type": "fixedValue", "value": [U_inlet, 0.0, 0.0]},
            "outlet":       {"type": "zeroGradient"},
            "top":          {"type": "slip"},   # FIX: was noSlip
            "bottom":       {"type": "slip"},   # FIX: was noSlip
            "cylinder":     {"type": "noSlip"},
            "frontAndBack": {"type": "symmetry"},
        }
    )
    write_foam_field(
        zero_dir / "p", "volScalarField",
        {"dimensions": "[0 2 -2 0 0 0 0]", "type": "uniform", "value": 0.0},
        {
            "inlet":        {"type": "zeroGradient"},
            "outlet":       {"type": "fixedValue", "value": "uniform 0.0"},
            "top":          {"type": "zeroGradient"},
            "bottom":       {"type": "zeroGradient"},
            "cylinder":     {"type": "zeroGradient"},
            "frontAndBack": {"type": "symmetry"},
        }
    )

    # 7. Run mesh and simulation
    case.run("blockMesh")
    case.run("snappyHexMesh -overwrite")

    # Diagnostic: find actual cylinder patch name from snappyHexMesh output.
    # snappyHexMesh derives patch names from the STL solid name / file stem,
    # but sometimes appends suffixes. Read the boundary file to get the truth.
    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
    cylinder_patch = "cylinder"   # default
    if boundary_file.exists():
        with open(boundary_file) as bf:
            btext = bf.read()
        # Find all patch names (words before a '{' block)
        patch_names = re.findall(r'^\s{4}(\w+)\s*$', btext, re.MULTILINE)
        print(f"\n    [diag] Patches in mesh: {patch_names}")
        # Pick the patch that contains 'cyl' or is not a standard name
        std = {'inlet', 'outlet', 'top', 'bottom', 'frontAndBack',
               'defaultFaces', 'empty', 'symmetry'}
        # Prefer any patch containing 'cyl'; fall back to first non-standard
        cyl_candidates = [p for p in patch_names
                          if p.lower() not in std and 'cyl' in p.lower()]
        if not cyl_candidates:
            cyl_candidates = [p for p in patch_names if p.lower() not in std]
        cylinder_patch = cyl_candidates[-1] if cyl_candidates else 'cylinder'
        print(f"    [diag] Using cylinder patch: {cylinder_patch!r}")
        # Rewrite the controlDict forces patches entry with the discovered name
        if boundary_file.exists():
            cd_path = system_dir / "controlDict"
            cd_text = cd_path.read_text()
            cd_text = re.sub(
                r'(patches\s+\()([^)]+)(\);)',
                rf'\g<1>{cylinder_patch}\g<3>',
                cd_text
            )
            cd_path.write_text(cd_text)

    case.run("simpleFoam")

    # 8. Extract Cd
    forces_file = find_forces_file(case_dir)
    fx, _, _ = parse_forces_file(forces_file)
    A_ref = diameter * channel_width
    Cd = 2.0 * fx / (1.0 * U_inlet**2 * A_ref)

    return Re, Cd

# (Place the functions here)

# ══════════════════════════════════════════════════════════════════════════════
#  Scaled batch section – replace the old if __name__ block
# ══════════════════════════════════════════════════════════════════════════════
def simulation_worker(args):
    """Runs one case and returns (dia, xc, Re, Cd)."""
    dia, xc, U_inlet, nu, target_cell_size = args
    Re, Cd = build_and_run_cylinder(
        dia, xc,
        U_inlet=U_inlet,
        nu=nu,
        target_cell_size=target_cell_size,
        base_dir="large_cylinder_cases"
    )
    return (dia, xc, Re, Cd)

if __name__ == "__main__":
    # ── Parameter grid ─────────────────────────────────────────────────
    diameters   = np.linspace(0.1, 0.5, 20)   # 20 diameters
    x_positions = np.linspace(1.0, 3.0, 20)   # 20 x‑positions

    # Add tiny random perturbation to avoid grid bias
    d_step = diameters[1] - diameters[0]
    x_step = x_positions[1] - x_positions[0]
    np.random.seed(42)
    diameters   += np.random.uniform(-0.025 * d_step, 0.025 * d_step, size=len(diameters))
    x_positions += np.random.uniform(-0.025 * x_step, 0.025 * x_step, size=len(x_positions))

    U_inlet          = 0.001
    nu               = 1e-5
    target_cell_size = 0.05

    total_cases = len(diameters) * len(x_positions)
    print(f"🚀 Launching {total_cases} simulations...")

    USE_PARALLEL = True
    NUM_WORKERS  = 6

    output_csv = Path("cylinder_dataset_large.csv")
    with open(output_csv, "w", newline="") as f:
        csv.writer(f).writerow(["diameter", "x_center", "Re", "Cd"])

    failed = []
    completed = 0
    start_time = time.time()

    if USE_PARALLEL and NUM_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {}
            for dia in diameters:
                for xc in x_positions:
                    args = (dia, xc, U_inlet, nu, target_cell_size)
                    fut = executor.submit(simulation_worker, args)
                    futures[fut] = (dia, xc)

            for fut in as_completed(futures):
                dia, xc = futures[fut]
                try:
                    dia, xc, Re, Cd = fut.result()
                    row = (round(dia, 6), round(xc, 6), round(Re, 2), round(Cd, 5))
                    with open(output_csv, "a", newline="") as f:
                        csv.writer(f).writerow(row)
                    completed += 1
                    elapsed = time.time() - start_time
                    est_total = elapsed / completed * total_cases
                    print(f"  [{completed}/{total_cases}]  D={dia:.3f}  x={xc:.3f}  "
                          f"Re={Re:.1f}  Cd={Cd:.4f}  (≈{est_total/60:.0f} min total)")
                except Exception as exc:
                    failed.append((dia, xc, str(exc)))
                    completed += 1
                    print(f"  [{completed}/{total_cases}]  D={dia:.3f}  x={xc:.3f}  ❌ FAILED: {exc}")
    else:
        for dia in diameters:
            for xc in x_positions:
                try:
                    dia, xc, Re, Cd = simulation_worker((dia, xc, U_inlet, nu, target_cell_size))
                    row = (round(dia, 6), round(xc, 6), round(Re, 2), round(Cd, 5))
                    with open(output_csv, "a", newline="") as f:
                        csv.writer(f).writerow(row)
                    completed += 1
                    elapsed = time.time() - start_time
                    est_total = elapsed / completed * total_cases
                    print(f"  [{completed}/{total_cases}]  D={dia:.3f}  x={xc:.3f}  "
                          f"Re={Re:.1f}  Cd={Cd:.4f}  (≈{est_total/60:.0f} min total)")
                except Exception as exc:
                    failed.append((dia, xc, str(exc)))
                    completed += 1
                    print(f"  [{completed}/{total_cases}]  D={dia:.3f}  x={xc:.3f}  ❌ FAILED: {exc}")

    total_time = time.time() - start_time
    print(f"\n✅  Dataset saved to {output_csv}")
    print(f"    {completed - len(failed)} / {total_cases} successful in {total_time/60:.1f} min")
    if failed:
        print(f"    ⚠️  {len(failed)} failed cases (see above).")