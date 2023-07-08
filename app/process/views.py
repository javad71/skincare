import os
import glob
import json
import psutil
import sys
import time
import numpy as np
import torch
import cv2
import random
import string
import requests
import torchvision.transforms as transforms
import torch.nn as nn

from skimage import io
from PIL import Image
from deepface.commons import functions
from deepface.extendedmodels import Age
from django.core.files.base import ContentFile
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone
from django.http import HttpResponse
from .apps import *
from .core.acne.fastrcnn_modules import nms_pytorch, fast_skin_quality_and_quantity
from .core.acne.face_detect import crop_face
from .core.acne.face_segment import face_segmentation
from .core.beauty.nets import ComboNet
from .core.general.general_modules import truncate_content
from .models import AcneQueue, Log


# use custom response class to override HttpResponse.close()
class LogSuccessResponse(HttpResponse):
    def close(self):
        super(LogSuccessResponse, self).close()
        # do whatever you want, this is the last codepoint in request handling
        continue_cycle()


class FacialBeautyPredictor:
    """
    Facial Beauty Predictor
    """

    def __init__(self, pretrained_model_path):
        model = ComboNet(num_out=5, backbone_net_name='SEResNeXt50')
        model = model.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        if torch.cuda.device_count() > 1:
            print("We are running on", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(pretrained_model_path))
        else:
            state_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        model.to(device)
        model.eval()

        self.device = device
        self.model = model

    def infer(self, img_file):
        tik = time.time()
        img = io.imread(img_file)
        img = Image.fromarray(img.astype(np.uint8))

        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = preprocess(img)
        img.unsqueeze_(0)
        img = img.to(self.device)

        score, cls = self.model(img)
        tok = time.time()

        return {
            'beauty': float(score.to('cpu').detach().item()),
            'elapse': tok - tik
        }


def beauty_score(pk):
    try:
        # read queue object from database
        try:
            queue_obj = AcneQueue.objects.get(id=pk)
        except AcneQueue.DoesNotExist:
            message = {'result': 'Queue object does not exist',
                       'code': '650',
                       'status': 4,
                       'beauty_stat': 500}
            return message

        # read image path from object
        img_file = str(queue_obj.image)
        # Create facial beauty predictor object
        fbp = FacialBeautyPredictor(pretrained_model_path='./process/core/beauty/ComboNet_SCUTFBP5500.pth')
        # inference to image
        result = fbp.infer(img_file)

        # log request
        Log.objects.create(request_time=timezone.now(),
                           request_id=pk,
                           request_event=f'calculate beauty score for request with key {pk}',
                           request_error_message={}, )
        set_status = {
            'status': 1,
            'beauty_stat': 200
        }
        result.update(set_status)

        return result

    except Exception as e:
        print('--->', e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        error = {
            'result': str(e),
            'line_of_error': str(exc_tb.tb_lineno),
            'code': '651',
            'status': 4,
            'beauty_stat': 500
        }

        # log request
        Log.objects.create(request_time=timezone.now(),
                           request_id=pk,
                           request_event='beauty_score() function have error',
                           request_error_message=error, )
        return error


def acne_score(pk):
    try:
        # log request
        Log.objects.create(request_time=timezone.now(),
                           request_id=pk,
                           request_event=f'enter request with key {pk}',
                           request_error_message={},)

        # read queue object from database
        try:
            queue_obj = AcneQueue.objects.get(id=pk)
        except AcneQueue.DoesNotExist:
            message = {'result': 'Queue object does not exist',
                       'code': '640',
                       'status': 4,
                       'acne_stat': 500}
            return message

        # read image path from object
        first_file = str(queue_obj.image)

        # main image
        main_img = cv2.imread(first_file)

        # detect skin and remove background
        media_path = "/usr/src/app/media/inputs/"
        tmp_file = face_segmentation(first_file, media_path, '.jpg')

        # define the detection threshold...
        # ... any detection having score below this will be discarded
        detection_threshold = 0.4
        non_maximum_suppression = 0.3

        # get the image file name for saving output later on
        image = cv2.imread(tmp_file)
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float64)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float)
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        # process image by pretrained model
        with torch.no_grad():
            start = time.time()
            global acne_model
            outputs = acne_model(image)
            end = time.time()
            delta = end - start
            print('process time for acne detection: ', delta, ' (s)')

        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        # apply non-maximum suppression
        outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels'] = nms_pytorch(outputs[0]['boxes'],
                                                                                      outputs[0]['scores'],
                                                                                      outputs[0]['labels'],
                                                                                      non_maximum_suppression)
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            labels = outputs[0]['labels'].data.numpy()

            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            labels = labels[scores >= detection_threshold].astype(np.int32)

            draw_boxes = boxes.copy()
            # get all the predicted class names
            point, white, black, papol, paschol, nedol, kist, quality = \
                fast_skin_quality_and_quantity(labels)

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(main_img,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 0, 255), 2)

            # Prepare directory
            outputs_file_dir = os.path.join(settings.MEDIA_ROOT, 'outputs')
            try:
                # Make directory
                oldmask = os.umask(000)
                os.makedirs(outputs_file_dir, mode=0o777)
            except OSError as e:
                if e.errno == 17:  # Dir already exists.
                    pass

            # saved process image by cv2
            media_path = "/usr/src/app/media/outputs/"
            image_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])
            image_extension = ".jpg"
            relative_uri = "".join([media_path, image_name, image_extension])
            cv2.imwrite(relative_uri, main_img)
            cv2.destroyAllWindows()

            # prepare image url
            relative_uri = relative_uri.replace('/usr/src/app/', '')
            image_url_output = os.path.join('https://ai.quta.ir/', relative_uri)

            # log request
            Log.objects.create(request_time=timezone.now(),
                               request_id=pk,
                               request_event=f'key {pk} exit with process (acne)',
                               request_error_message={}, )

            result = {
                'status': 1,
                'imageUrl': image_url_output,
                'point': point,
                'quality skin': quality,
                'white acne detected': white,
                'black acne detected': black,
                'papol acne detected': papol,
                'paschol acne detected': paschol,
                'nedol acne detected': nedol,
                'kist acne detected': kist,
                'acne_stat': 200
            }
            return result
        else:
            first_file = first_file.replace('/usr/src/app/', '')
            image_url_output = os.path.join('https://ai.quta.ir/', first_file)

            # log request
            Log.objects.create(request_time=timezone.now(),
                               request_id=pk,
                               request_event=f'key {pk} exit without process (acne)',
                               request_error_message={}, )

            result = {
                'status': 1,
                'imageUrl': image_url_output,
                'point': 100,
                'quality skin': 'خیلی خوب',
                'white acne detected': 0,
                'black acne detected': 0,
                'papol acne detected': 0,
                'paschol acne detected': 0,
                'nedol acne detected': 0,
                'kist acne detected': 0,
                'acne_stat': 200
            }
            return result

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        error = {
            'result': str(e),
            'line_of_error': str(exc_tb.tb_lineno),
            'code': '642',
            'status': 4,
            'acne_stat': 500
        }

        # log request
        Log.objects.create(request_time=timezone.now(),
                           request_id=pk,
                           request_event='acne_score() function have error',
                           request_error_message=error, )
        return error


def detect_identity(pk):
    try:
        # read queue object from database
        try:
            queue_obj = AcneQueue.objects.get(id=pk)
        except AcneQueue.DoesNotExist:
            message = {'result': 'Queue object does not exist',
                       'code': '690',
                       'status': 4,
                       'identity_stat': 500}
            return message

        # read image path from object
        tmp_file = str(queue_obj.image)

        start = time.time()
        # ------------------------------------------------------------------------------------------
        global age_model, race_model
        face_224 = functions.preprocess_face(img=tmp_file, target_size=(224, 224), grayscale=False,
                                             enforce_detection=False, detector_backend='opencv')
        age_predictions = age_model.predict(face_224)[0, :]
        apparent_age = Age.findApparentAge(age_predictions)

        race_predictions = race_model.predict(face_224)[0, :]
        race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
        sum_of_predictions = race_predictions.sum()
        race_obj = {"race": {}}
        for i in range(0, len(race_labels)):
            race_label = race_labels[i]
            race_prediction = 100 * race_predictions[i] / sum_of_predictions
            race_obj["race"][race_label] = race_prediction

        race_obj["dominant_race"] = race_labels[np.argmax(race_predictions)]

        if race_obj["dominant_race"] == race_labels[0]:
            race = 'نژاد آسیایی'
        elif race_obj["dominant_race"] == race_labels[1]:
            race = 'نژاد هندی'
        elif race_obj["dominant_race"] == race_labels[2]:
            race = 'نژاد آفریقایی'
        elif race_obj["dominant_race"] == race_labels[3]:
            race = 'نژاد اروپایی'
        elif race_obj["dominant_race"] == race_labels[4]:
            race = 'نژاد خاورمیانه ای'
        else:
            race = 'نژاد آمریکای لاتین'

        result = {'age': apparent_age, 'race': race, 'status': 1, 'identity_stat': 200}
        # ------------------------------------------------------------------------------------------
        end = time.time()
        delta = end - start
        print('process time for predict age and race: ', delta, ' (s)')

        # log request
        Log.objects.create(request_time=timezone.now(),
                           request_id=pk,
                           request_event=f'predict age and race for request with key {pk}',
                           request_error_message={}, )

        return result

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        error = {
            'result': str(e),
            'line_of_error': str(exc_tb.tb_lineno),
            'code': '691',
            'status': 4,
            'identity_stat': 500
        }

        # log request
        Log.objects.create(request_time=timezone.now(),
                           request_id=pk,
                           request_event='detect_identity() function have error',
                           request_error_message=error, )
        return error


def continue_cycle():
    try:
        queryset = AcneQueue.objects.filter(status__lt=2)
        if len(queryset) > 0:
            print('start processing...')
            reader_process()
        else:
            print('system is idle...')
            resend_process()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        error = {
            'result': str(e),
            'line_of_error': str(exc_tb.tb_lineno),
            'code': '600'
        }
        Log.objects.create(request_time = timezone.now(),
                           request_id = 0,
                           request_event = 'continue_cycle() function have error',
                           request_error_message = error,)


def resend_process():
    try:
        if AcneQueue.objects.filter(status=4).exists():
            queryset = AcneQueue.objects.filter(status=4)
            for process in queryset:
                if process.time_to_dead < 3:
                    print('resend data to php api')
                    requests.post('https://api.quta.ir/api/v1/analizRes', data=process.result)
                    process.time_to_dead = process.time_to_dead + 1
                    process.save()
                else:
                    print('terminate process after three unsuccessful attempts')
                    process.status = 3
                    process.save()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        error = {
            'result': str(e),
            'line_of_error': str(exc_tb.tb_lineno),
            'code': '610'
        }
        Log.objects.create(request_time=timezone.now(),
                           request_id=0,
                           request_event='resend_process() function have error',
                           request_error_message=error, )


def beat(index):
    try:
        # read first record from RequestTable based on time enqueue
        selected_image = AcneQueue.objects.filter(status__lt=2).order_by('time_enqueue')[index]

        # read primary key of selected record
        pk = getattr(selected_image, 'id')

        # update time_dequeue and status of selected record
        AcneQueue.objects.filter(id=pk).update(time_dequeue=timezone.now(), status=2)

        # Run API modules
        result = {
            'private_key': settings.PRIVATE_KEY,
            'id': pk,
        }

        # define variables
        result_acne = None
        result_beauty = None
        result_identity = None

        # read image file
        image_file = getattr(selected_image, 'image')

        # assess if image contain face or not
        temp, valid_image_flag = crop_face(str(image_file))
        if valid_image_flag:
            # read model number of selected record
            model = getattr(selected_image, 'model')
            for key, item in model.items():
                if key == 'acne_score':
                    if item:
                        # detect acne with FastRCNN
                        result_acne = acne_score(pk)
                        result.update(result_acne)
                if key == 'beauty_score':
                    if item:
                        # predict beauty score using squeeze and excite network
                        result_beauty = beauty_score(pk)
                        result.update(result_beauty)
                if key == 'age_race':
                    if item:
                        # predict age and race with deepface
                        result_identity = detect_identity(pk)
                        result.update(result_identity)
                if key == 'skin_disease':
                    if item:
                        # detect skin disease
                        result_skin_disease = {}
                        result.update(result_skin_disease)

            try:
                if result_acne is None and 'acne_score' in model:   
                    set_status = {'status': 4}
                    result.update(set_status)

                if result_beauty is None and 'beauty_score' in model:   
                    set_status = {'status': 4}
                    result.update(set_status)

                if result_identity is None and 'age_race' in model:
                    set_status = {'status': 4}
                    result.update(set_status)     
                
                if 'acne_score' in model: 
                    if result_acne['acne_stat'] == 500:
                        set_status = {'status': 4}
                        result.update(set_status)
                
                if 'beauty_score' in model:
                    if result_beauty['beauty_stat'] == 500:
                        set_status = {'status': 4}
                        result.update(set_status)
                
                if 'age_race' in model: 
                    if result_identity['identity_stat'] == 500:
                        set_status = {'status': 4}
                        result.update(set_status)
            except Exception as e:
                print('check results has error: ', str(e))

            # save result of process
            AcneQueue.objects.filter(id=pk).update(result=result)

            # call dequeue_process
            dequeue_process(result=result, pk=pk)
        else:
            set_status = {'status': 4}
            result.update(set_status)

            # save result of process
            AcneQueue.objects.filter(id=pk).update(result=result)

            # call dequeue_process
            dequeue_process(result=result, pk=pk)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        error = {
            'result': str(e),
            'line_of_error': str(exc_tb.tb_lineno),
            'code': '630'
        }
        Log.objects.create(request_time=timezone.now(),
                           request_id=0,
                           request_event='beat() function have error',
                           request_error_message=error, )


def reader_process():
    try:
        # time for delete process from the queue if a model can't process it
        deadline = 120 # seconds
        # If queue is not empty then start
        if AcneQueue.objects.all().exists():
            # calculate number of running process
            n_running_process = len(AcneQueue.objects.filter(status=2))
            if n_running_process < 1:
                beat(index=0)
            else:
                running_process = AcneQueue.objects.filter(status=2)
                for process in running_process:
                    if (timezone.now() - process.time_dequeue).seconds > deadline:
                        pk = process.id
                        # log event
                        Log.objects.create(request_time=timezone.now(),
                                           request_id=pk,
                                           request_event='deadline is over...',
                                           request_error_message={}, )
                        # terminate process from queue and update status and time dequeue
                        AcneQueue.objects.filter(id=pk).update(time_dequeue=timezone.now(), status=3)
                        post_data = {
                            'private_key': settings.PRIVATE_KEY,
                            'id': pk,
                            'status': 3,
                            'imageUrl': 'unknown',
                            'point': 0,
                            'quality skin': 'unknown',
                            'white acne detected': 0,
                            'black acne detected': 0,
                            'papol acne detected': 0,
                            'paschol acne detected': 0,
                            'nedol acne detected': 0,
                            'kist acne detected': 0,
                            'code': status.HTTP_500_INTERNAL_SERVER_ERROR,
                        }
                        response = requests.post('https://api.quta.ir/api/v1/analizRes', data=post_data)
                        content = response.content
                        print('php api response -->', content)

    except Exception as e:
        # log request
        exc_type, exc_obj, exc_tb = sys.exc_info()
        error = {
            'result': str(e),
            'line_of_error': str(exc_tb.tb_lineno),
            'code': '620'
        }
        Log.objects.create(request_time=timezone.now(),
                           request_id=0,
                           request_event='reader_process() function have error',
                           request_error_message=error, )

        # Call reader_process function for continue cycle
        continue_cycle()


def dequeue_process(result, pk):
    try:
        # send message to php api
        print('result ', result)
        response = requests.post('https://api.quta.ir/api/v1/analizRes', data=result)
        if response is None or response.status_code != 200:
            AcneQueue.objects.filter(id=pk).update(status=4)
        else:
            # terminate request (Dequeue)
            AcneQueue.objects.filter(id=pk).update(time_dequeue=timezone.now(), status=3)

        print('RAM memory % used:', psutil.virtual_memory()[2])

        # Call reader_process function for continue cycle
        continue_cycle()

    except Exception as e:
        # log request
        exc_type, exc_obj, exc_tb = sys.exc_info()
        error = {
            'result': str(e),
            'line_of_error': str(exc_tb.tb_lineno),
            'code': '660'
        }
        Log.objects.create(request_time=timezone.now(),
                           request_id=pk,
                           request_event='dequeue_process() function have error',
                           request_error_message=error, )

        # Call reader_process function for continue cycle
        continue_cycle()


@api_view(['POST'])
@csrf_exempt
def grabbing(request):
    image_file = request.FILES["image_file"]
    private_key = request.POST.get('private_key', None)

    model = request.POST.get('model', None)
    if model is None:
        model = {
            'acne_score': True
        }

    if private_key == settings.PRIVATE_KEY:
        if image_file:
            try:
                # Read file and saved it
                image_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])
                path = default_storage.save(f'inputs/{image_name}.jpg', ContentFile(image_file.read()))
                tmp_file = os.path.join(settings.MEDIA_ROOT, path)

                # save image in database (enqueue)
                acne_object = AcneQueue.objects.create(image=tmp_file,
                                                       time_enqueue=timezone.now(),
                                                       time_dequeue=None,
                                                       status=1,
                                                       model=json.loads(model),
                                                       time_to_dead=0,
                                                       result={})
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                error = {
                    'result': str(e),
                    'line_of_error': str(exc_tb.tb_lineno),
                    'code': '500'
                }
                return Response(error, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            error = {
                'result': 'image_file must be sent.',
                'code': '400'
            }
            return Response(error, status=status.HTTP_400_BAD_REQUEST)
    else:
        error = {
            'result': 'The request passed to the service is not valid',
            'code': '400'
        }
        return Response(error, status=status.HTTP_400_BAD_REQUEST)

    content = {
        'id': acne_object.id,
        'status': '200'
    }
    response = LogSuccessResponse(HttpResponse(json.dumps(content), content_type="application/json"))
    return response


@api_view(['GET'])
@csrf_exempt
def get_queue_status(request):
    queryset = AcneQueue.objects.all()
    result = {
        'number of image in queue': len(queryset),
        'data': queryset.values()
    }
    return Response(result, status=status.HTTP_200_OK)


@api_view(['POST'])
@csrf_exempt
def get_log_by_date(request):
    year = request.POST.get('year', None)
    month = request.POST.get('month', None)
    day = request.POST.get('day', None)
    if year is not None and month is not None and day is not None:
        queryset = Log.objects.filter(request_time__year=year,
                                      request_time__month=month,
                                      request_time__day=day)
    else:
        error = {'data': 'year, month and day must be sent...',
                 'code': '670'}
        return Response(error, status=status.HTTP_400_BAD_REQUEST)

    result = {
        'data': queryset.values(),
        'code': '200'
    }
    return Response(result, status=status.HTTP_200_OK)


@api_view(['POST'])
@csrf_exempt
def get_log_by_id(request):
    pk = request.POST.get('pk', None)
    try:
        queryset = Log.objects.filter(request_id=pk)
    except Log.DoesNotExist:
        error = {'data': 'Log object does not exist',
                 'code': '680'}
        return Response(error, status=status.HTTP_400_BAD_REQUEST)

    result = {
        'data': queryset.values(),
        'code': '200'
    }
    return Response(result, status=status.HTTP_200_OK)


@api_view(['GET'])
@csrf_exempt
def truncate_queue(request):
    try:
        AcneQueue.objects.all().delete()
    except:
        return Response('cant truncate table...', status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return Response('table is truncated successfully', status=status.HTTP_200_OK)


@api_view(['GET'])
@csrf_exempt
def get_statistic_volumes(request):
    try:
        # path joining version for other paths
        inputs_dir = '/usr/src/app/media/inputs/'
        outputs_dir = '/usr/src/app/media/outputs/'
        n_files_inputs = len([name for name in os.listdir(inputs_dir) if os.path.isfile(os.path.join(inputs_dir, name))])
        n_files_outputs = len([name for name in os.listdir(outputs_dir) if os.path.isfile(os.path.join(outputs_dir, name))])
        size_inputs_dir = round(sum(d.stat().st_size for d in os.scandir(inputs_dir) if d.is_file())/(1024**2))
        size_outputs_dir = round(sum(d.stat().st_size for d in os.scandir(outputs_dir) if d.is_file())/(1024**2))
        result = {
            'number_of_files_inputs': n_files_inputs,
            'number_of_files_outputs': n_files_outputs,
            'size_inputs_dir': str(size_inputs_dir) + ' (MB)',
            'size_outputs_dir': str(size_outputs_dir) + ' (MB)',
        }
        return Response(result, status=status.HTTP_200_OK)
    except Exception as e:
        return Response(str(e), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@csrf_exempt
def delete_contents_volumes(request):
    inputs_dir = '/usr/src/app/media/inputs/'
    outputs_dir = '/usr/src/app/media/outputs/'
    result1 = truncate_content(folder=inputs_dir)
    result2 = truncate_content(folder=outputs_dir)
    if result1 & result2:
        return Response('truncation is done ...', status=status.HTTP_200_OK)
    else:
        return Response('cant truncate directory ...', status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# call continue cycle function for restart service situation
try:
    continue_cycle()
except:
    pass
