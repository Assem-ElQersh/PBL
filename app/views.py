from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from app import util
@csrf_exempt
def upload_video(request):
    if request.method == 'POST':
        video_file = request.FILES['video']
        # Handle the video file (e.g., save it to a directory)
        with open('media/uploaded_videos/video.mp4', 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        x = util.DataLoader.DataLoader.load_inference_data('media/uploaded_videos/video.mp4')
        print(x.shape)
        prediction = util.predict(x)
        print(prediction)
        return JsonResponse({'status': 'success', 'message' : prediction})
    
    return JsonResponse({'status': 'failed', 'message' : 'No prediction'})

def index(request):
    return render(request, 'app/index.html')
