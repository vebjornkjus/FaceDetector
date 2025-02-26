import cv2
import pygame
import sys
import time

# --- Konfigurasjon ---
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1200
ANIMATION_SPEED = 6.5        # Øker for raskere respons, senkes for glattere bevegelse
MOVEMENT_SENSITIVITY = 5     # Juster hvor mye pupillene beveger seg
# ----------------------

# Initialiser webkamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kunne ikke åpne webkameraet.")
    sys.exit()

# Last inn Haar Cascade for ansiktsgjenkjenning
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("Kunne ikke laste Haar Cascade. Sjekk banen til XML-filen.")
    sys.exit()

# Initialiser Pygame
pygame.init()
display_info = pygame.display.Info()
WINDOW_WIDTH = display_info.current_w
WINDOW_HEIGHT = display_info.current_h
screen = pygame.display.set_mode(
    (WINDOW_WIDTH, WINDOW_HEIGHT),
    pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
)
pygame.display.set_caption("Øyne som følger ansiktet")
clock = pygame.time.Clock()

# Definer øyeposisjoner og størrelser
left_eye_center = (WINDOW_WIDTH // 4, WINDOW_HEIGHT // 2)
right_eye_center = (3 * WINDOW_WIDTH // 4, WINDOW_HEIGHT // 2)
eye_radius = min(WINDOW_WIDTH, WINDOW_HEIGHT) // 8
pupil_radius = eye_radius // 4
max_pupil_movement = eye_radius // 5  # Maks bevegelse for pupillen

# Hent en start-ramme for å finne kameraets oppløsning
ret, frame = cap.read()
if not ret:
    print("Feil ved henting av ramme.")
    sys.exit()
frame_height, frame_width = frame.shape[:2]

# Initialiser animasjonsvariabler
current_dx, current_dy = 0, 0
target_dx, target_dy = 0, 0
last_time = time.time()

def get_target_face(gray_frame):
    """Returnerer det største ansiktet i bildet, eller None hvis ingen ansikt er funnet."""
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    target = None
    max_area = 0
    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            target = (x, y, w, h)
    return target

def lerp(current, target, dt, speed):
    """Lineær interpolering med tidsavhengig faktor."""
    lerp_factor = min(1.0, speed * dt)
    return current + (target - current) * lerp_factor

try:
    while True:
        # Håndter Pygame-hendelser
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt

        # Hent et bilde fra webkameraet
        ret, frame = cap.read()
        if not ret:
            continue

        # Konverter bildet til gråskala for ansiktsgjenkjenning
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        target_face = get_target_face(gray)

        if target_face is not None:
            x, y, w, h = target_face
            face_center_x = x + w / 2
            face_center_y = y + h / 2
            # Kalkuler relativ posisjon (-1 til 1)
            rel_x = (face_center_x - frame_width / 2) / (frame_width / 2)
            rel_y = (face_center_y - frame_height / 2) / (frame_height / 2)
            
            target_dx = -rel_x * max_pupil_movement * MOVEMENT_SENSITIVITY
            target_dy = rel_y * max_pupil_movement * MOVEMENT_SENSITIVITY
            
            # Skaler bevegelsen basert på ansiktsstørrelse (simulerer avstand)
            movement_scale = 1.0 - min(0.8, (w * h) / (frame_width * frame_height))
            target_dx *= movement_scale
            target_dy *= movement_scale
        else:
            target_dx, target_dy = 0, 0

        # Beregn delta tid for jevn animasjon
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # Interpoler pupillens posisjon mot målverdien
        current_dx = lerp(current_dx, target_dx, dt, ANIMATION_SPEED)
        current_dy = lerp(current_dy, target_dy, dt, ANIMATION_SPEED)

        # Beregn pupillposisjoner
        left_pupil_center = (int(left_eye_center[0] + current_dx), int(left_eye_center[1] + current_dy))
        right_pupil_center = (int(right_eye_center[0] + current_dx), int(right_eye_center[1] + current_dy))

        # Tegn bakgrunn, øyne og pupiller
        screen.fill((0, 0, 0))  # svart bakgrunn
        pygame.draw.circle(screen, (255, 255, 255), left_eye_center, eye_radius)
        pygame.draw.circle(screen, (255, 255, 255), right_eye_center, eye_radius)
        pygame.draw.circle(screen, (0, 0, 0), left_pupil_center, pupil_radius)
        pygame.draw.circle(screen, (0, 0, 0), right_pupil_center, pupil_radius)

        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    # Rydd opp før avslutning
    cap.release()
    pygame.quit()
    sys.exit()