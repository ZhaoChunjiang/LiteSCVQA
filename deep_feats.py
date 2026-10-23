"""Lightweight CNN (MobileNetV3-Small, ONNX, CPU) features + BRISQUE baseline features for MSU videos.
Usage: python deep_feats.py out.csv [start] [end]   (content folder range)
"""
import sys, os, time, glob, numpy as np, cv2, onnx, onnxruntime as ort
from scipy.special import gamma as G

NF = 8                      # frames per video, uniformly spaced
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def load_session(path='models/mnv3s.onnx', layer='/global_pool/pool/GlobalAveragePool_output_0'):
    m = onnx.load(path)
    m.graph.output.insert(0, onnx.helper.make_tensor_value_info(layer, onnx.TensorProto.FLOAT, None))
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    return ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider']), layer

def prep(rgb):
    x = (rgb.astype(np.float32) / 255. - MEAN) / STD
    return x.transpose(2, 0, 1)[None]

# ---------- BRISQUE (Mittal et al. 2012) ----------
GAM = np.arange(0.03, 10, 0.001); R_GGD = G(1 / GAM) * G(3 / GAM) / G(2 / GAM) ** 2
def ggd(x):
    s2 = np.mean(x ** 2); r = s2 / (np.mean(np.abs(x)) ** 2 + 1e-12)
    return GAM[np.argmin((R_GGD - r) ** 2)], s2
def aggd(x):
    l = x[x < 0]; r = x[x > 0]
    sl = np.sqrt(np.mean(l ** 2)) if l.size else 1e-6; sr = np.sqrt(np.mean(r ** 2)) if r.size else 1e-6
    gh = sl / sr; rh = np.mean(np.abs(x)) ** 2 / (np.mean(x ** 2) + 1e-12)
    Rh = rh * (gh ** 3 + 1) * (gh + 1) / (gh ** 2 + 1) ** 2
    a = GAM[np.argmin((R_GGD ** -1 - Rh) ** 2)] if False else GAM[np.argmin((G(2 / GAM) ** 2 / (G(1 / GAM) * G(3 / GAM)) - Rh) ** 2)]
    const = np.sqrt(G(1 / a) / G(3 / a)); bl = sl * const; br = sr * const
    mean = (br - bl) * G(2 / a) / G(1 / a)
    return [a, mean, sl ** 2, sr ** 2]
def mscn(g):
    mu = cv2.GaussianBlur(g, (7, 7), 7 / 6); s = np.sqrt(np.abs(cv2.GaussianBlur(g * g, (7, 7), 7 / 6) - mu * mu))
    return (g - mu) / (s + 1)
def brisque(gray):
    f = []; g = gray.astype(np.float64)
    for _ in range(2):
        m = mscn(g); f += list(ggd(m))
        for sh in [m[:, 1:] * m[:, :-1], m[1:, :] * m[:-1, :], m[1:, 1:] * m[:-1, :-1], m[1:, :-1] * m[:-1, 1:]]:
            f += aggd(sh.ravel())
        g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
    return f

def video_feats(path, sess, layer):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = set(np.linspace(0, n - 1, NF + 2)[1:-1].astype(int)); frames = []; i = 0
    while True:
        ok = cap.grab()
        if not ok: break
        if i in idx:
            ok, f = cap.retrieve(); frames.append(f)
        i += 1
    A, B, Q = [], [], []
    for f in frames:
        h, w = f.shape[:2]; rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        s = 512 / w; small = cv2.resize(rgb, (512, int(round(h * s / 32)) * 32), interpolation=cv2.INTER_AREA)
        A.append(sess.run([layer], {'pixel_values': prep(small)})[0].ravel())          # global (downscaled) view
        c = 384; crops = [rgb[y:y + c, x:x + c] for y in (h // 4 - c // 2, 3 * h // 4 - c // 2) for x in (w // 4 - c // 2, 3 * w // 4 - c // 2)]
        crops = [cr for cr in crops if cr.shape[:2] == (c, c)]
        B.append(sess.run([layer], {'pixel_values': np.concatenate([prep(cr) for cr in crops])})[0].reshape(len(crops), -1).mean(0))  # native-res detail view
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); Q.append(brisque(cv2.resize(gray, (768, int(h * 768 / w)), interpolation=cv2.INTER_AREA)))
    A, B, Q = np.array(A), np.array(B), np.array(Q)
    return np.concatenate([A.mean(0), A.std(0), B.mean(0)]), Q.mean(0)

def main():
    out = sys.argv[1]; a = int(sys.argv[2]) if len(sys.argv) > 2 else 1; b = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    sess, layer = load_session(); rows = []
    done = set()
    if os.path.exists(out):
        done = set(l.split(',')[0] for l in open(out).read().splitlines()[1:])
    fh = open(out, 'a')
    if not done:
        fh.write('video,' + ','.join([f'cg_mean{i}' for i in range(576)] + [f'cg_std{i}' for i in range(576)] + [f'cl_mean{i}' for i in range(576)] + [f'brisque{i}' for i in range(36)] + ['sec']) + '\n')
    for k in range(a, b + 1):
        for v in sorted(glob.glob(f'msu/{k}/*.mp4')):
            if v.endswith('orig.mp4'): continue
            key = f'{k}/{os.path.basename(v)}'
            if key in done: continue
            t = time.time(); c, q = video_feats(v, sess, layer); dt = time.time() - t
            fh.write(key + ',' + ','.join(f'{x:.6g}' for x in np.concatenate([c, q])) + f',{dt:.3f}\n'); fh.flush()
        print('done content', k, flush=True)

if __name__ == '__main__':
    main()
