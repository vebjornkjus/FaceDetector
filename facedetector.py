import cv2
import pygame
import sys
import time
import math  # Nytt: for mikro-bevegelse

# --- Konfigurasjon ---
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1200
ANIMATION_SPEED = 6.5        # Øker for raskere respons, senkes for glattere bevegelse
MOVEMENT_SENSITIVITY = 5     # Juster hvor mye pupillene beveger seg
# ----------------------

# Nytt: Variabler for blink og stabilitet
blink_duration = 0.2       # Blink varer i 0.2 sekunder
blink_active = False       # Er øynene i ferd med å blinke
blink_start_time = 0
no_face_time = 0           # Hvor lenge siden et ansikt sist ble funnet
flicker_threshold = 0.5    # Terskel (i sekunder) for når blink skal aktiveres ved fravær

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

# Initialiser Pygame og hent faktisk skjermoppløsning
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
left_eye_center = (WINDOW_WIDTH // 3, WINDOW_HEIGHT // 2)      # Justert for bedre sentrering
right_eye_center = (2 * WINDOW_WIDTH // 3, WINDOW_HEIGHT // 2)
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

        # Hent gjeldende tid og beregn delta-tid
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        if target_face is not None:
            # Ansikt funnet – oppdater posisjon og nullstill "ingen ansikt"-timer
            no_face_time = 0
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

            # Fjern mikro-bevegelsen fra pupill-offset her
            # (Mikro-bevegelsen skal nå kun gjelde øyene)
        else:
            # Ingen ansikt funnet – øk timeren, men hold forrige target hvis vi er innenfor terskelen
            no_face_time += dt
            if no_face_time >= flicker_threshold:
                target_dx, target_dy = 0, 0
                if not blink_active:
                    blink_active = True
                    blink_start_time = current_time

        # Interpoler pupillens posisjon mot målverdien
        current_dx = lerp(current_dx, target_dx, dt, ANIMATION_SPEED)
        current_dy = lerp(current_dy, target_dy, dt, ANIMATION_SPEED)

        # Legg til en liten mikro-bevegelse for øynene (den hvite sirkelen)
        eye_micro_dx = math.sin(current_time * 1.5) * (max_pupil_movement * 0.2)
        eye_micro_dy = math.cos(current_time * 1.5) * (max_pupil_movement * 0.2)

        # Beregn dynamiske øyeposisjoner
        dynamic_left_eye_center = (left_eye_center[0] + eye_micro_dx, left_eye_center[1] + eye_micro_dy)
        dynamic_right_eye_center = (right_eye_center[0] + eye_micro_dx, right_eye_center[1] + eye_micro_dy)

        # Beregn pupillposisjoner relativt til de dynamiske øyeposisjonene
        left_pupil_center = (int(dynamic_left_eye_center[0] + current_dx), int(dynamic_left_eye_center[1] + current_dy))
        right_pupil_center = (int(dynamic_right_eye_center[0] + current_dx), int(dynamic_right_eye_center[1] + current_dy))

        screen.fill((0, 0, 0))  # svart bakgrunn

        if blink_active:
            if current_time - blink_start_time < blink_duration:
                # Tegn "lukkede" øyne med dynamiske posisjoner
                pygame.draw.line(screen, (0, 0, 0), 
                                 (dynamic_left_eye_center[0] - eye_radius, dynamic_left_eye_center[1]), 
                                 (dynamic_left_eye_center[0] + eye_radius, dynamic_left_eye_center[1]), 
                                 eye_radius // 3)
                pygame.draw.line(screen, (0, 0, 0), 
                                 (dynamic_right_eye_center[0] - eye_radius, dynamic_right_eye_center[1]), 
                                 (dynamic_right_eye_center[0] + eye_radius, dynamic_right_eye_center[1]), 
                                 eye_radius // 3)
            else:
                blink_active = False
                pygame.draw.circle(screen, (255, 255, 255), dynamic_left_eye_center, eye_radius)
                pygame.draw.circle(screen, (255, 255, 255), dynamic_right_eye_center, eye_radius)
                pygame.draw.circle(screen, (0, 0, 0), left_pupil_center, pupil_radius)
                pygame.draw.circle(screen, (0, 0, 0), right_pupil_center, pupil_radius)
        else:
            # Normal tegning med dynamiske øyeposisjoner
            pygame.draw.circle(screen, (255, 255, 255), dynamic_left_eye_center, eye_radius)
            pygame.draw.circle(screen, (255, 255, 255), dynamic_right_eye_center, eye_radius)
            pygame.draw.circle(screen, (0, 0, 0), left_pupil_center, pupil_radius)
            pygame.draw.circle(screen, (0, 0, 0), right_pupil_center, pupil_radius)

        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    # Rydd opp før avslutning
    cap.release()
    pygame.quit()
    sys.exit()