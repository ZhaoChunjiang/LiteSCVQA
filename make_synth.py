# Synthetic smoke-test dataset: 10 sources x 8 distortions, pseudo-MOS from known distortion levels.
import cv2, numpy as np, os, subprocess, csv
W,H,fps,SEC=960,540,25,5
rng=np.random.default_rng(1)
words="signal image video filter sampling fourier wavelet entropy matrix vector gradient kernel".split()
def slide(src,k):
    r=np.random.default_rng(src*100+k)
    bg=tuple(int(v) for v in r.integers(200,256,3)); fg=tuple(int(v) for v in r.integers(0,80,3))
    img=np.full((H,W,3),bg,np.uint8)
    sc=r.uniform(0.5,0.9)
    cv2.putText(img,f"Chapter {src}.{k}",(40,70),cv2.FONT_HERSHEY_SIMPLEX,sc*1.8,fg,2)
    for i in range(7):
        t=" ".join(r.choice(words,5))
        cv2.putText(img,f"- {t}",(50,130+i*45),cv2.FONT_HERSHEY_SIMPLEX,sc,fg,2 if sc>0.7 else 1)
    if src%2==0:
        pic=cv2.GaussianBlur(r.integers(0,255,(150,220,3),dtype=np.uint8),(0,0),4)
        img[360:510,700:920]=pic
    return img
def frames_for(src):
    fr=[]; per=fps*SEC//3
    for k in range(3):
        b=slide(src,k)
        for t in range(per):
            f=b.copy(); cv2.circle(f,(100+t*6,330),6,(0,0,255),-1); fr.append(f)
    return fr
def write(path,fr):
    vw=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
    for f in fr: vw.write(f)
    vw.release()
def stall(fr,period,length):
    out=[];i=0
    while len(out)<len(fr):
        if i%period==5 and length>0: out+=[fr[min(i,len(fr)-1)]]*length; i+=length
        else: out.append(fr[min(i,len(fr)-1)]); i+=1
    return out[:len(fr)]
D=[("crf18",dict(crf=18)),("crf30",dict(crf=30)),("crf38",dict(crf=38)),("crf45",dict(crf=45)),
   ("down2",dict(crf=18,scale=2)),("down4",dict(crf=18,scale=4)),("stall_s",dict(crf=18,stall=(40,6))),("stall_l",dict(crf=18,stall=(25,12)))]
rows=[]
os.makedirs("synth",exist_ok=True)
for src in range(10):
    fr=frames_for(src); d=f"synth/{src:03d}"; os.makedirs(d,exist_ok=True)
    for name,p in D:
        f2=stall(fr,*p["stall"]) if "stall" in p else fr
        write("tmp.mp4",f2)
        vf=[]
        if "scale" in p: s=p["scale"]; vf=["-vf",f"scale={W//s}:{H//s},scale={W}:{H}"]
        subprocess.run(["ffmpeg","-loglevel","error","-y","-i","tmp.mp4",*vf,"-c:v","libx264","-crf",str(p["crf"]),f"{d}/{name}.mp4"],check=True)
        q=100-1.6*max(0,p["crf"]-18)-{2:12,4:35}.get(p.get("scale"),0)
        if "stall" in p: q-= 300*p["stall"][1]/p["stall"][0]/3
        rows.append([f"{src:03d}/{name}.mp4",round(q+rng.normal(0,3),2)])
os.remove("tmp.mp4")
with open("synth_mos.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["video","mos"]); w.writerows(rows)
print(len(rows),"videos")
