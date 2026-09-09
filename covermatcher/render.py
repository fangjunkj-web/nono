from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

def cover_crop(im,size):
    w,h=size; s=max(w/im.width,h/im.height); nw,nh=int(im.width*s),int(im.height*s)
    im=im.resize((nw,nh),Image.Resampling.LANCZOS)
    x=(nw-w)//2; y=(nh-h)//2
    return im.crop((x,y,x+w,y+h))

def font(size,bold=False):
    candidates=['/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf','/System/Library/Fonts/SFNS.ttf']
    for p in candidates:
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def wrap(draw,text,f,maxw):
    words=text.split(); lines=[]; cur=''
    for word in words:
        test=(cur+' '+word).strip()
        if draw.textbbox((0,0),test,font=f)[2] <= maxw: cur=test
        else:
            if cur: lines.append(cur)
            cur=word
    if cur: lines.append(cur)
    return lines[:5]

def render(image_path,title,out_path,logo_path=None,size=(1200,630),accent=(0,103,255)):
    im=cover_crop(Image.open(image_path).convert('RGB'),size)
    overlay=Image.new('RGBA',size,(0,0,0,0)); d=ImageDraw.Draw(overlay)
    d.rectangle((0,0,int(size[0]*.58),size[1]),fill=(4,18,35,185))
    for x in range(int(size[0]*.58),int(size[0]*.78)):
        a=int(185*(1-(x-size[0]*.58)/(size[0]*.20)))
        d.line((x,0,x,size[1]),fill=(4,18,35,max(0,a)))
    im=Image.alpha_composite(im.convert('RGBA'),overlay)
    d=ImageDraw.Draw(im)
    if logo_path and Path(logo_path).exists():
        lg=Image.open(logo_path).convert('RGBA'); lg.thumbnail((180,85),Image.Resampling.LANCZOS); im.alpha_composite(lg,(55,42))
    f=font(50,True); lines=wrap(d,title,f,530); y=190
    for line in lines:
        d.text((58,y),line,font=f,fill='white'); y+=60
    d.rounded_rectangle((58,min(y+20,555),150,min(y+28,563)),4,fill=accent+(255,))
    d.text((58,575),'AI COVER MATCHER',font=font(18,True),fill=(220,230,240,255))
    Path(out_path).parent.mkdir(parents=True,exist_ok=True); im.convert('RGB').save(out_path,quality=94,optimize=True)
