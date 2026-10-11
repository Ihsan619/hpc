"""Reproducible validation inputs; assignment implementations remain C/CUDA."""
from pathlib import Path
import hashlib
import json
import random
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

def encrypt(raw):
    values = [ord(raw[0])+2, ord(raw[0])-2, ord(raw[0])+1,
              ord(raw[1])+3, ord(raw[1])-3, ord(raw[1])-1,
              ord(raw[2])+2, ord(raw[2])-2, ord(raw[3])+4, ord(raw[3])-4]
    for i, value in enumerate(values):
        low, high = (97, 122) if i < 6 else (48, 57)
        if value > high:
            values[i] = value - high + low
        elif value < low:
            values[i] = low - value + low
    return ''.join(map(chr, values))

def main():
    rng = random.Random(619)
    raw = ['aa00', 'aa99', 'zz00', 'zz99', 'az09', 'za90']
    for index in rng.sample(range(67600), 9994):
        raw.append(f'{chr(97+index//2600)}{chr(97+index//100%26)}{index//10%10}{index%10}')
    (ROOT/'task3/data/passwords.txt').write_text('\n'.join(map(encrypt, raw))+'\n', encoding='ascii')
    (ROOT/'validation/passwords_expected.txt').write_text('\n'.join(raw)+'\n', encoding='ascii')
    (ROOT/'validation/words_boundary.txt').write_text('Alpha alpha ALPHA, beta! gamma\n'+ 'z'*8193 +' beta\nlast', encoding='ascii')
    (ROOT/'validation/matrices_cases.txt').write_text(
        '2,2\n1,2\n3,4\n\n2,2\n0,2\n-3,4\n\n'
        '2,3\n1,2,3\n4,5,6\n\n3,2\n7,8\n9,10\n11,12\n\n'
        '1,3\n2,3,4\n\n2,1\n5\n6\n', encoding='ascii')
    image = Image.new('RGBA', (512,384), (25,40,60,255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40,50,470,335), fill=(190,175,150,255))
    for x in range(70,450,60):
        for y in range(85,310,55):
            draw.rectangle((x,y,x+30,y+30), fill=(35,70,110,255))
    draw.polygon([(15,50),(255,10),(495,50)], fill=(115,90,75,255))
    draw.line((0,370,511,345), fill=(230,230,230,255), width=5)
    image.save(ROOT/'task4/data/building.png')
    for name,size in [('tiny',(1,1)),('thin',(1,7)),('partial_block',(17,19))]:
        image = Image.new('RGBA', size)
        image.putdata([((x*41)%256,(y*61)%256,((x+y)*17)%256,(x*13+y*23)%256)
                       for y in range(size[1]) for x in range(size[0])])
        image.save(ROOT/f'task4/data/{name}.png')
    files = sorted([*ROOT.glob('task*/data/*'), *ROOT.glob('validation/*_expected.txt')])
    manifest = {p.relative_to(ROOT).as_posix(): {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    (ROOT/'docs/input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Generated seed-619 passwords, edge cases, synthetic PNGs, and SHA-256 manifest.')

if __name__ == '__main__':
    main()
