from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
from PIL import Image

# Load pre-trained models
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Define a transform to normalize the image data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Function to perform face recognition
def recognize_face(face_img):
    # Detect faces
    boxes, probs = mtcnn.detect(face_img)

    # If no faces are detected, return None
    if boxes is None:
        return "Unknown"

    # Get face embeddings
    faces = [transform(Image.fromarray(face_img).crop(box)) for box in boxes]
    embeddings = torch.stack([resnet(face.unsqueeze(0)).detach() for face in faces])

    # Perform face recognition (for demonstration, just return "Known" if a face is detected)
    return "Known"
