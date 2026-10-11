"""Create Colab notebooks that run pinned C/CUDA sources from public GitHub."""
from pathlib import Path
import json
import subprocess
ROOT = Path(__file__).resolve().parents[1]
REVISION = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

def markdown(text): return {'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}
def code(text): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(True)}

titles = ['Word occurrence counting with Pthreads','Matrix operations with OpenMP',
          'Password cracking with CUDA','Sobel filtering across multiple PNGs with CUDA']
builds = ['task1/word_count','task2/matrix_operations','task3/password_cracking','task4/sobel']
commands = ["['./word_count','data/WordOccurrenceDataset.txt','4']",
            "['./matrix_operations','data/MatData.txt','4']",
            "['./password_cracking','data/passwords.txt']",
            "['./sobel','edges',*map(str,sorted(Path('data').glob('*.png')))]"]
for n in range(1,5):
    task=f'task{n}'; filename=f'Task_{n:02d}.ipynb'
    cells = [markdown(f'# Task {n}: {titles[n-1]}\n\n6CS005 High Performance Computing. '
                      'The assessed implementation is the separate C/CUDA source in the public repository. '
                      'Python cells only download, compile, execute, validate, and display evidence.\n\n'
                      + ('Select **Runtime > Change runtime type > T4 GPU** before running.\n\n' if n>=3 else '')
                      + 'Run the cells in order. Inputs are public; no personal Google Drive is mounted. '
                      'Record actual execution on your PC after all checks pass.\n'),
             markdown('## 1. Fetch the exact source and input revision\n'),
             code(f'''import os, sys, json, subprocess, hashlib, platform
from pathlib import Path
REVISION = "{REVISION}"
ROOT = Path("/content/hpc_" + REVISION[:12])
if not ROOT.exists():
    subprocess.run(["git","clone","--no-checkout","https://github.com/Ihsan619/hpc.git",str(ROOT)],check=True)
    subprocess.run(["git","checkout",REVISION],cwd=ROOT,check=True)
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip() == REVISION
os.chdir(ROOT)
manifest = json.loads((ROOT/"docs/input_manifest.json").read_text())
for name,record in manifest.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == record["sha256"], name
print("Public source revision:", REVISION)
print("Input SHA-256 checks passed:", len(manifest))
environment = {{"revision":REVISION,"platform":platform.platform(),"python":sys.version}}
for command in [["gcc","--version"],["lscpu"],["nvcc","--version"],["nvidia-smi"]]:
    try:
        result=subprocess.run(command,text=True,capture_output=True)
        environment[" ".join(command)] = result.stdout + result.stderr
        print(environment[" ".join(command)])
    except FileNotFoundError:
        environment[" ".join(command)] = "unavailable"
(ROOT/"evidence").mkdir(exist_ok=True)
(ROOT/"evidence/{task}_environment.json").write_text(json.dumps(environment,indent=2))
'''),
             markdown('## 2. Compile the standalone source\n'),
             code(("subprocess.run(['nvidia-smi'],check=True)\n" if n>=3 else '')+
                  f"subprocess.run(['make','{builds[n-1]}'],cwd=ROOT,check=True)\n"),
             markdown('## 3. Run the task and inspect its output\n'),
             code(f'''os.chdir(ROOT/"{task}")
command = {commands[n-1]}
print("Command:", " ".join(command))
result = subprocess.run(command,text=True,capture_output=True,check=True)
print(result.stdout)
(ROOT/"evidence/{task}_demonstration.txt").write_text(result.stdout)
''')]
    if n<4:
        output=['result.txt','results.txt','decrypted.txt'][n-1]
        cells.append(code(f'''output = Path("{output}").read_text().splitlines()
print("Output file: {output} | Lines:",len(output))
print("\\n".join(output[:45]))
'''))
    else:
        cells.append(code('os.chdir(ROOT/"task4")\nfrom PIL import Image\nimport matplotlib.pyplot as plt\npath = Path(\'data/bird.png\')\nfig, axes = plt.subplots(2, 2, figsize=(10, 10))\nfig.suptitle(path.name)\nviews = [(path, \'Original Image\'),\n         (Path(\'edges\')/f\'gradient_x_{path.name}\', \'Gradient in X direction\'),\n         (Path(\'edges\')/f\'gradient_y_{path.name}\', \'Gradient in Y direction\'),\n         (Path(\'edges\')/f\'sobel_{path.name}\', \'Sobel Edge Detection\')]\nfor axis, (image_path, title) in zip(axes.flat, views):\n    axis.imshow(Image.open(image_path).convert(\'RGBA\')); axis.set_title(title); axis.axis(\'off\')\nplt.tight_layout()\nfig.savefig(ROOT/\'evidence/task4_bird_comparison.png\', dpi=160, bbox_inches=\'tight\')\nplt.show()\n'))
    cells += [markdown('## 4. Independent correctness checks and repeated timing trials\n\n'
                       'Validation checks normal and invalid inputs. Three measured trials follow '
                       'one excluded warm-up per configuration. Process wall time includes I/O and '
                       'startup; the C/CUDA console reports the narrower compute/kernel timing. '
                       'Do not describe these as interchangeable measurements.\n'),
              code(f'''os.chdir(ROOT)
subprocess.run([sys.executable,"validation/validate.py","{task}","--benchmark"],check=True)
record = json.loads((ROOT/"evidence/{task}_validation.json").read_text())
assert record["passed"]
print("Independent verification: PASS")
import statistics
groups = {{}}
for trial in record["benchmark_trials"]:
    groups.setdefault(trial["threads"],[]).append(trial["process_wall_seconds"])
for threads,values in groups.items():
    print("Threads:",threads,"| process wall median:",round(statistics.median(values),6),"s | trials:",values)
'''),
              markdown('## 5. Download actual outputs and evidence\n\n'
                       'This downloads generated files, validation records, and runtime details. '
                       'Keep screen recordings separately under the local project evidence folder.\n'),
              code(f'''import zipfile
bundle=Path('/content/{task}_outputs_and_evidence.zip')
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in (ROOT/'evidence').glob('{task}_*'):
        archive.write(path,path.relative_to(ROOT))
    {'for path in (ROOT/"task4/edges").glob("*.png"): archive.write(path,path.relative_to(ROOT))' if n==4 else f'archive.write(ROOT/"{task}/{output}","{task}/{output}")'}
print("Bundle:",bundle)
from google.colab import files
files.download(str(bundle))
''')]
    notebook={'nbformat':4,'nbformat_minor':5,
              'metadata':{'colab':{'name':filename},'kernelspec':{'display_name':'Python 3','name':'python3'},
                          **({'accelerator':'GPU'} if n>=3 else {})},'cells':cells}
    (ROOT/'notebooks'/filename).write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')
print('Created four notebooks pinned to',REVISION)
