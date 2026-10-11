"""Independent reference checks for the four C/CUDA executables.
Run from the repository root: python3 validation/validate.py task1 [--benchmark]
Each program runs in a temporary directory, avoiding output-file collisions.
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import math
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
RECORDS = []

def run(exe, args, directory, expected=0):
    command = [str(ROOT/exe), *map(str,args)]
    start = time.perf_counter()
    result = subprocess.run(command,cwd=directory,text=True,capture_output=True,timeout=240)
    elapsed = time.perf_counter()-start
    RECORDS.append({'command':[str(x) for x in command], 'returncode':result.returncode,
                    'wall_seconds':elapsed,'stdout':result.stdout,'stderr':result.stderr})
    assert result.returncode == expected, f'{command}\n{result.stdout}\n{result.stderr}'
    return result, elapsed

def count_reference(path):
    return Counter(re.findall(r'[a-z0-9]+',path.read_text(encoding='ascii').lower()))

def read_counts(path):
    lines = path.read_text().splitlines()
    pairs = [line.rsplit(' ',1) for line in lines]
    assert [p[0] for p in pairs] == sorted(p[0] for p in pairs)
    assert len({p[0] for p in pairs}) == len(pairs)
    return {word:int(number) for word,number in pairs}

def words(directory):
    for path in [ROOT/'task1/data/WordOccurrenceDataset.txt', ROOT/'validation/words_boundary.txt']:
        expected = count_reference(path)
        for threads in [1,2,4,16]:
            run('task1/word_count',[path,threads],directory)
            assert read_counts(directory/'result.txt') == expected
    empty = directory/'empty.txt'; empty.write_text('')
    run('task1/word_count',[empty,4],directory)
    assert (directory/'result.txt').read_text() == ''
    for invalid in ['0','-1','abc','2x','1025']:
        run('task1/word_count',[empty,invalid],directory,1)
    run('task1/word_count',[directory/'missing.txt',4],directory,1)

def parse_matrices(path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    result = []; index = 0
    while index < len(lines):
        rows,cols = map(int,lines[index].split(',')); index += 1
        matrix = [list(map(float,lines[index+r].split(','))) for r in range(rows)]
        assert all(len(row)==cols for row in matrix)
        index += rows; result.append(matrix)
    return result

def matrix_reference(a,b):
    ar,ac,br,bc = len(a),len(a[0]),len(b),len(b[0])
    result = {}
    if (ar,ac)==(br,bc):
        result['Addition'] = [[a[i][j]+b[i][j] for j in range(ac)] for i in range(ar)]
        result['Subtraction'] = [[a[i][j]-b[i][j] for j in range(ac)] for i in range(ar)]
        result['Element-wise multiplication'] = [[a[i][j]*b[i][j] for j in range(ac)] for i in range(ar)]
        result['Element-wise division'] = [[a[i][j]/b[i][j] if b[i][j] else math.nan for j in range(ac)] for i in range(ar)]
    result['Transpose A'] = list(map(list,zip(*a)))
    result['Transpose B'] = list(map(list,zip(*b)))
    if ac == br:
        result['Matrix multiplication'] = [[sum(a[i][k]*b[k][j] for k in range(ac)) for j in range(bc)] for i in range(ar)]
    return result

def check_matrix_output(path, matrices):
    lines = path.read_text().splitlines(); current = 0; seen = set()
    expected = None; pair_count = 0
    while current < len(lines):
        line = lines[current]; current += 1
        if not line: continue
        if line.startswith('Pair '):
            if expected is not None: assert seen == set(expected)
            pair_count += 1; seen = set()
            assert line.startswith(f'Pair {pair_count} |')
            expected = matrix_reference(*matrices[(pair_count-1)*2:pair_count*2])
        elif ' cannot be done ' in line:
            name = line.split(' cannot be done ')[0]; assert name not in expected
        else:
            name,shape = line.split(' - '); rows,cols = map(int,shape.split(','))
            assert name in expected and name not in seen
            reference = expected[name]; assert (rows,cols)==(len(reference),len(reference[0]))
            actual = [list(map(float,lines[current+i].split(','))) for i in range(rows)]
            current += rows; seen.add(name)
            for values,correct in zip(actual,reference):
                assert len(values)==cols
                for value,wanted in zip(values,correct):
                    assert (math.isnan(value) and math.isnan(wanted)) or math.isclose(value,wanted,rel_tol=1e-11,abs_tol=1e-9),(name,value,wanted)
    assert seen == set(expected) and pair_count == len(matrices)//2

def matrices(directory):
    for path in [ROOT/'task2/data/MatData.txt',ROOT/'validation/matrices_cases.txt']:
        reference = parse_matrices(path)
        for threads in [1,2,4,32]:
            run('task2/matrix_operations',[path,threads],directory)
            check_matrix_output(directory/'results.txt',reference)
    malformed = ['','x,2\n1,2\n','0,2\n','2,2\n1,2\n','1,2\n1,x\n',
                 '1,2\n1,2,3\n','1,2\n1\n2\n','1,1\n1\n','1,1\nNaN\n','1,1\n1,\n']
    for text in malformed:
        path = directory/'bad.txt'; path.write_text(text)
        run('task2/matrix_operations',[path,4],directory,1)
    run('task2/matrix_operations',[directory/'missing.txt',4],directory,1)
    run('task2/matrix_operations',[ROOT/'validation/matrices_cases.txt','0'],directory,1)

def passwords(directory):
    from generate_fixtures import encrypt
    run('task3/password_cracking',[ROOT/'task3/data/passwords.txt'],directory)
    recovered = (directory/'decrypted.txt').read_text().splitlines()
    expected = (ROOT/'validation/passwords_expected.txt').read_text().splitlines()
    assert recovered == expected
    assert list(map(encrypt,recovered)) == (ROOT/'task3/data/passwords.txt').read_text().splitlines()
    # A single final line without newline and an impossible but well-formed cipher.
    path = directory/'single.txt'; path.write_text(encrypt('zz99'))
    run('task3/password_cracking',[path],directory)
    assert (directory/'decrypted.txt').read_text() == 'zz99\n'
    path.write_text('aaaaaaaaaa'); run('task3/password_cracking',[path],directory,1)
    path.write_text('aaaaaa0000'); run('task3/password_cracking',[path],directory,2)
    assert (directory/'decrypted.txt').read_text() == 'NOT_FOUND\n'
    path.write_text(''); run('task3/password_cracking',[path],directory,1)
    run('task3/password_cracking',[directory/'missing.txt'],directory,1)

def sobel_reference(image):
    import numpy as np
    pixels = np.asarray(image.convert('RGBA'),dtype=np.int32)
    gray = (30*pixels[:,:,0]+59*pixels[:,:,1]+11*pixels[:,:,2])//100
    padded = np.pad(gray,1,mode='constant'); h,w = gray.shape
    gx = np.zeros((h,w),dtype=np.int32); gy = gx.copy()
    kx = [[-1,0,1],[-2,0,2],[-1,0,1]]; ky = [[-1,-2,-1],[0,0,0],[1,2,1]]
    for y in range(3):
        for x in range(3):
            window = padded[y:y+h,x:x+w]
            gx += kx[y][x]*window; gy += ky[y][x]*window
    edge = np.minimum(255,np.floor(np.sqrt(gx.astype(float)**2+gy.astype(float)**2))).astype(np.uint8)
    outputs = {}
    for prefix, values in [('sobel_', edge), ('gradient_x_', np.minimum(255,np.abs(gx))), ('gradient_y_', np.minimum(255,np.abs(gy)))]:
        output = np.asarray(image.convert('RGBA')).copy()
        output[:,:,:3] = values[:,:,None]
        if prefix != 'sobel_': output[:,:,3] = 255
        outputs[prefix] = output
    return outputs

def images(directory):
    from PIL import Image
    import numpy as np
    paths = sorted((ROOT/'task4/data').glob('*.png'))
    output = directory/'edges'
    run('task4/sobel',[output,*paths],directory)
    for path in paths:
        with Image.open(path) as image: expected = sobel_reference(image)
        for prefix, reference in expected.items():
            with Image.open(output/f'{prefix}{path.name}') as image: actual = np.asarray(image.convert('RGBA'))
            assert np.array_equal(actual,reference), f'Pixel mismatch: {prefix}{path.name}'
    run('task4/sobel',[output,directory/'missing.png'],directory,1)
    run('task4/sobel',[output,paths[0],paths[0]],directory,1)
    bad = directory/'broken.png'; bad.write_bytes(b'not a PNG')
    run('task4/sobel',[output,bad],directory,1)

def benchmark(task,directory):
    trials = []
    if task=='task1':
        source = ROOT/'task1/data/WordOccurrenceDataset.txt'
        large = directory/'words_large.txt'; large.write_bytes(source.read_bytes()*10)
        configs = [('task1/word_count',[large,n],n) for n in [1,2,4,8]]
    elif task=='task2':
        large = directory/'matrix_large.txt'; rng = __import__('random').Random(619)
        with large.open('w') as stream:
            for _ in range(2):
                stream.write('384,384\n')
                for _ in range(384): stream.write(','.join(str(rng.randrange(1,100)) for _ in range(384))+'\n')
                stream.write('\n')
        configs = [('task2/matrix_operations',[large,n],n) for n in [1,2,4,8]]
    elif task=='task3':
        configs = [('task3/password_cracking',[ROOT/'task3/data/passwords.txt'],None)]
    else:
        configs = [('task4/sobel',[directory/'bench_edges',ROOT/'task4/data/bird.png'],None)]
    for exe,args,threads in configs:
        run(exe,args,directory) # warm-up, excluded from reported trials
        for trial in range(3):
            result,elapsed = run(exe,args,directory)
            trials.append({'threads':threads,'trial':trial+1,'process_wall_seconds':elapsed,'stdout':result.stdout})
    return trials

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('task',choices=['task1','task2','task3','task4'])
    parser.add_argument('--benchmark',action='store_true'); options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='hpc_validation_') as name:
        directory = Path(name)
        {'task1':words,'task2':matrices,'task3':passwords,'task4':images}[options.task](directory)
        trials = benchmark(options.task,directory) if options.benchmark else []
    output = ROOT/'evidence'/f'{options.task}_validation.json'; output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({'task':options.task,'passed':True,'checks':len(RECORDS),'runs':RECORDS,'benchmark_trials':trials},indent=2))
    print(f'{options.task}: PASS | {len(RECORDS)} program runs | independent reference checks passed')
    print(f'Validation record: {output}')

if __name__=='__main__':
    main()
