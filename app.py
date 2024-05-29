import cv2
import numpy as np
import pyaudio
import colorsys

# Initialize PyAudio
p = pyaudio.PyAudio()

# Sound parameters
RATE = 44100  # Sample rate
DURATION = 0.1  # Duration of the sound in seconds

# Function to generate a sine wave sound
def generate_tone(frequency, duration, rate=RATE):
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    return wave.astype(np.float32).tobytes()

# Function to determine the dominant color in an image
def get_dominant_color(image):
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    dominant_color = palette[0]
    return dominant_color

# Function to convert RGB color to a frequency
def color_to_frequency(color):
    # Convert RGB to HSV
    r, g, b = color
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    frequency = 440 + (h * 880)  # Map hue to a frequency range (A4 to A5)
    return frequency

# Function to play sound
def play_sound(frequency):
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=RATE,
                    output=True)
    sound = generate_tone(frequency, DURATION)
    stream.write(sound)
    stream.stop_stream()
    stream.close()

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame to make processing faster
        resized_frame = cv2.resize(frame, (100, 100))

        # Get the dominant color
        dominant_color = get_dominant_color(resized_frame)
        print(f"Dominant Color: {dominant_color}")

        # Convert dominant color to frequency
        frequency = color_to_frequency(dominant_color)

        # Play sound based on the dominant color
        play_sound(frequency)

        # Display the frame
        cv2.imshow('frame', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    p.terminate()
