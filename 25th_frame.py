import tkinter as tk
import tkinter.messagebox
import cv2 as cv
from PIL import Image, ImageTk
from collections import deque
import numpy as np
import time
import sys
import os
import subprocess as sp

def current_time():
    return round(time.time() * 1000)

def mse(img1, img2):
   h, w, _ = img1.shape
   diff = cv.absdiff(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse, diff

def draw_text(img, text,
          pos=(0, 0),
          font=cv.FONT_HERSHEY_PLAIN,
          font_scale=3,
          font_thickness=2,
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, (x, y - 10), (x + text_w, y + text_h + 30), text_color_bg, -1)
    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def format_time(frame_i, fps):
    time = (frame_i+1) // fps
    return f'{int(time // 3600)}:{int((time // 60) % 60):02d}:{int(time % 60):02d}.{int((frame_i+1) % fps)}'

def add_timestamp(frame, i, fps):
    h, w, _ = frame.shape
    draw_text(frame, format_time(i, fps), (int(h / 10), int(w / 10)), cv.FONT_HERSHEY_SIMPLEX, max(1, int(h / 200)), 6)

def resize(frame, width):
    h, w, _ = frame.shape
    return cv.resize(frame, (int(width), int(h * width / w)))

def show(capture, frames):
    pos = 0
    cnt = len(frames)
    fps = capture.get(cv.CAP_PROP_FPS)
    patched = [False] * cnt
    should_patch = False

    window = tk.Tk()
    window.resizable(False, False)
    width = window.winfo_screenwidth() / 3.5
    
    aspect_ratio = capture.get(cv.CAP_PROP_FRAME_WIDTH) / capture.get(cv.CAP_PROP_FRAME_HEIGHT)

    def update_image():
        images = []
        capture.set(cv.CAP_PROP_POS_FRAMES, frames[pos])
        for i in range(3):
            success, frame = capture.read()
            if not success:
                break
            images.append(frame)
        if patched[pos]:
            images[1] = images[0].copy()
        for i in range(3):
            add_timestamp(images[i], frames[pos] + i, fps)
        img = Image.fromarray(cv.cvtColor(cv.hconcat(images), cv.COLOR_BGR2RGB)).resize(
            (int(width * 3), int(width / aspect_ratio)))
        img = ImageTk.PhotoImage(image = img)
        image['image'] = img
        image.image = img
        window.title(f'Frames preview: {pos+1}/{cnt}')

    def command_prev():
        nonlocal pos
        if pos == 0:
            return
        pos -= 1
        if pos == 0:
            button_prev['state'] = 'disabled'
        button_next['state'] = 'normal'
        update_image()

    def command_next():
        nonlocal pos
        if pos == cnt - 1:
            return
        pos += 1
        if pos == cnt - 1:
            button_next['state'] = 'disabled'
        button_prev['state'] = 'normal'
        update_image()

    def command_preview():
        nonlocal capture
        i = frames[pos] - 20
        capture.set(cv.CAP_PROP_POS_FRAMES, max(i, 0))
        images = []
        while i < frames[pos] + 20:
            success, frame = capture.read()
            if not success:
                break
            images.append(frame)
            i += 1
        if patched[pos]:
            images[21] = images[20].copy()
        i = frames[pos] - 20
        for frame in images:
            add_timestamp(frame, i, fps)
            frame = resize(frame, width * 3)
            cv.imshow('Preview', frame)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
            time.sleep(0.1)
            i += 1
        cv.destroyAllWindows()
    
    def command_patch():
        patched[pos] = True
        update_image()

    def command_finish():
        nonlocal should_patch
        #if any(patched):
        #    should_patch = tkinter.messagebox.askquestion(
        #        'Question', 'Do you want to patch selected frames?',
        #        icon='question'
        #        ) == 'yes'
        window.destroy()

    button_prev = tk.Button(
        text="Previous",
        width=7,
        height=1,
        command=command_prev,
        state='disabled'
    )
    button_preview = tk.Button(
        text="View",
        width=7,
        height=1,
        command=command_preview
    )
    button_patch = tk.Button(
        text="Patch",
        width=7,
        height=1,
        command=command_patch
    )
    button_finish = tk.Button(
        text="Finish",
        width=7,
        height=1,
        command=command_finish
    )
    button_next = tk.Button(
        text="Next",
        width=7,
        height=1,
        command=command_next,
        state=('disabled' if cnt == 1 else 'normal')
    )

    image = tk.Label()
    update_image()
    image.pack()
    button_prev.pack(side=tk.LEFT)
    button_preview.pack(side=tk.LEFT)
    #button_patch.pack(side=tk.LEFT)
    button_next.pack(side=tk.LEFT)
    button_finish.pack(side=tk.RIGHT)

    window.mainloop()
    return [frames[i] + 1 for i in filter(lambda f: patched[f], range(cnt))], should_patch

def main():
    filename = input("Enter file path\n")
    while not os.path.isfile(filename):
        filename = input("No such file, try again\n")

    capture = cv.VideoCapture(filename)

    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv.CAP_PROP_FPS)

    buffer = deque()
    i = 0
    broken_frames = []
    try:
        f = open(filename + '-frames.txt', 'r')
        try:
            broken_frames.extend(int(i) for i in f.readlines())
            i = broken_frames[-1]
            capture.set(cv.CAP_PROP_POS_FRAMES, i)
        finally:
            f.close()
    except Exception:
        pass
    
    if len(broken_frames) == 0 or broken_frames[-1] != -1:
        deltatime = 0
        speed = 1
        deltaframes = 0
        f = open(filename + '-frames.txt', 'a')
        while True:
            start_time = current_time()
            success, frame = capture.read()
            if not success:
                f.write('-1\n')
                break
            
            buffer.append(frame)
            if len(buffer) == 4:
                buffer.popleft()
                diff01 = mse(buffer[0], buffer[1])[0]
                if diff01 > 100:
                    diff02 = mse(buffer[0], buffer[2])[0]
                    if diff01 - diff02 > diff01 / 3:
                        f.write(f'{i-2}\n')
                        broken_frames.append(i-2)
            
            dtime = current_time() - start_time
            deltatime += dtime
            deltaframes += 1
            if deltatime > 2000:
                speed = deltaframes / fps * 1000 / deltatime
                deltatime = 0
                deltaframes = 0

            print(f'Frame={i}/{frame_count} time={format_time(i, fps)} speed={speed:.2f}x ETA={format_time((frame_count-i) / speed, fps)} broken={len(broken_frames)}  ', end='\r')
            i += 1
        f.close()
        print()

    if len(broken_frames) > 0 and broken_frames[-1] == -1:
        broken_frames = broken_frames[:-1]

    if len(broken_frames) == 0:
        print("No broken frames!")
        capture.release()
        return
    patched_frames, should_patch = show(capture, broken_frames)
    if len(patched_frames) > 0 and should_patch:
        ffmpegCommand = ['ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', f'{int(capture.get(cv.CAP_PROP_FRAME_WIDTH))}x{int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-vcodec', 'h264',
        'out-' + filename ]

        proc = sp.Popen(ffmpegCommand, stdin=sp.PIPE)

        if sys.platform == "win32":
            import msvcrt
            msvcrt.setmode(proc.stdin.fileno(), os.O_BINARY)

        i = 0
        capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        while True:
            start_time = current_time()
            success, frame = capture.read()
            if not success:
                break
            if not i in patched_frames:
                out_frame = frame
            proc.stdin.write(out_frame.tobytes())
            i += 1
        proc.stdin.close()
        proc.stderr.close()
        proc.wait()
    elif len(patched_frames) > 0:
        pass
    else:
        #print("No frames to patch, exiting")
        pass
    capture.release()

if __name__ == "__main__":
    main()