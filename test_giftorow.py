from PIL import Image


frames_image = []

# GIF-Datei laden
gif_path = '/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/cor_t1/cor_pd_fs/classifier_gif_7cor_pd_fs.gif'
gif_image = Image.open(gif_path)

# Anzahl der Frames im GIF abrufen
num_frames = gif_image.n_frames

iterator = int(num_frames/8)

# Extrahieren und speichern der einzelnen Bilder
frame = 0
while frame < num_frames:

    if (frame + iterator) > num_frames:
        frame = num_frames-1

    # Einzelnes Bild im GIF auswählen
    gif_image.seek(frame)

    frames_image.append(gif_image.copy())

    frame+= iterator



#-------- schicker plot bilderreihe ---------
breite, höhe = frames_image[0].size
ausgabe = Image.new('RGBA', (breite * len(frames_image), höhe))
for index, bild in enumerate(frames_image):
    ausgabe.paste(bild, (index * breite, 0))
ausgabe.save('/home/yv312705/Code/diffusion_autoenc/eval_plots/mri_nine/gif_to_row/t1_pdfs_4.png')