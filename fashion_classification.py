import torch
import sys
import uvicorn
import aiohttp
import base64 
from PIL import Image as PILImage

from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai import *
from fastai.vision import *
from pathlib import Path
from io import BytesIO


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


def encode(img):
    img = (image2np(img.data) * 255).astype('uint8')
    pil_img = PILImage.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


app = Starlette()

# path = Path("/tmp")
# data = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
# learner = create_cnn(data, models.resnet34)
# learner.model.load_state_dict(
#     torch.load("model-weights.pth", map_location="cpu")
# )

defaults.device = torch.device('cpu')
path = Path("./")
learner = load_learner(path, fname='export_unfreeze_693.pkl')


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    pred_class, pred_idx, outputs = learner.predict(img)
    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(
            zip(learner.data.classes, map(str, formatted_outputs)),
            key=lambda p: p[1],
            reverse=False
        )
    img_data = encode(img)
    return HTMLResponse(
        """
        <html>
           <body>
             <p>Prediction: <b>%s</b></p>
             <p>Confidence: %s</p>
             <p>Top 3 predictions: %s</p>
           </body>
        <figure class="figure">
          <img src="data:image/png;base64, %s" class="figure-img img-thumbnail input-image">
        </figure>
        </html>
    """ %(pred_probs[0][0], pred_probs[0][1], pred_probs[:3], img_data))


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <h1>Clothes Classification Model !</h1>
        <p>We can classify any type of the cloth. Just upload an image.</p><br>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <u>Select image to upload:</u><br><p>
            1. <input type="file" name="file"><br><p>
            2. <input type="submit" value="Upload and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008))
        uvicorn.run(app, host="0.0.0.0", port=port)
