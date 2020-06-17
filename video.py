import sys
import cv2
import torch
import pafy
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from train import Model
import face_recognition


@torch.no_grad()
def detect(model_path, video_path):
    model = Model()
    state_dict = torch.load(model_path)['state_dict']
    model.load_state_dict(state_dict, strict=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('detector', cv2.WINDOW_NORMAL)
    labels = ['Without mask', 'With mask']
    label_color = [(0, 0, 255), (0, 255, 0)]

    vPafy = pafy.new(video_path)
    play = vPafy.getbest(preftype="mp4")
    vc = cv2.VideoCapture(play.url)

    scale = 4

    while True:
        (rVal, frame) = vc.read()
        dims = frame.shape
        min_img = cv2.resize(frame, (dims[1] // scale, dims[0] // scale))
        faces = face_recognition.face_locations(min_img)
        for face in faces:
            (top, right, bottom, left) = [v*scale for v in face]

            face_img = frame[top:bottom, left:right]
            output = model(transformations(face_img).unsqueeze(0).to(device))
            _, predicted = torch.max(output.data, 1)

            cv2.rectangle(frame, (left, top), (right, bottom), label_color[predicted], 2)
            cv2.putText(frame, labels[predicted], (left, top - 10), font, 0.6, label_color[predicted], 2)

        cv2.imshow('detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Wrong number of arguments.')
    detect(sys.argv[1], sys.argv[2])
