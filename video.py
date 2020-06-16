import click
import cv2
import torch
import pafy
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from common.facedetector import FaceDetector
from train import MaskDetector


@click.command(help="""
                    modelPath: path to model.ckpt\n
                    videoPath: path to video file to annotate
                    """)
@click.argument('model_path')
@click.argument('video_path')
@torch.no_grad()
def detect(model_path, video_path):
    """ detect if persons in video are wearing masks or not
    """
    model = MaskDetector()
    model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    face_detector = FaceDetector(
        prototype='models/deploy.prototxt.txt',
        model='models/res10_300x300_ssd_iter_140000.caffemodel',
    )
    
    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('detector', cv2.WINDOW_NORMAL)
    labels = ['Without mask', 'With mask']
    label_color = [(0, 0, 255), (0, 255, 0)]

    # vPafy = pafy.new(video_path)
    # play = vPafy.getbest(preftype="mp4")
    vc = cv2.VideoCapture(video_path)

    while True:
        (rVal, frame) = vc.read()
        faces = face_detector.detect(frame)
        for face in faces:
            (x, y, w, h) = face
            
            # clamp coordinates that are outside of the image
            x, y = max(x, 0), max(y, 0)
            
            # predict mask label on extracted face
            face_img = frame[y:y+h, x:x+w]
            output = model(transformations(face_img).unsqueeze(0).to(device))
            _, predicted = torch.max(output.data, 1)
            
            # draw face frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), label_color[predicted], 2)
            
            # draw prediction label
            cv2.putText(frame, labels[predicted], (x, y-20), font, 0.6, label_color[predicted], 2)

        cv2.imshow('detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect()
