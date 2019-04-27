import os
import sys
import time
from random import shuffle
from tqdm import tqdm
from graph import Graph
from imutils import url_to_image
from os.path import join as join_path
from instagram import Instagram
from faceutil import Detector


def shuffle_list(a, limit):
    shuffle(a)
    if len(a) > limit:
        a = a[:limit]
    return a


def log_event(status, text):
    unixtime = int(time.time())
    print "[{} {:>5}] {}".format(unixtime, status.upper(), text)


if __name__ == "__main__":
    limit_users = 200

    detector = Detector(
        join_path('models', 'shape_predictor_68_face_landmarks.dat'),
        join_path('models', 'dlib_face_recognition_resnet_model_v1.dat'),
        join_path('models', 'gender.pickle'),
    )
    graph = Graph(os.getenv('INSTAGRAM_NEO4J'))

    instagram = Instagram(
        os.getenv('INSTAGRAM_USERNAME'),
        os.getenv('INSTAGRAM_PASSWORD'),
    )
    current_user = os.getenv('INSTAGRAM_USERNAME')
    current_gender = 'male'

    log_event('info', 'trying to connect to Instagram ...')
    if not instagram.login():
        log_event('error', "couldn't connect to Instagram !")
        sys.exit(1)

    log_event('info', 'fetching logged-in user followers ...')
    followers = instagram.followers(instagram.username_id)
    if followers['status'] != 'ok':
        log_event('error', 'invalid Instagram status ' + followers['status'])
        sys.exit(2)

    for follower in shuffle_list(followers['users'], limit=limit_users):
        username = follower['username']
        username_id = follower['pk']

        try:
            image = url_to_image(follower['profile_pic_url'])
            gender = detector.process(image)
            follower['gender'] = gender
            graph.add_edge(username, gender, current_user, current_gender)
        except Exception as e:
            log_event('warn', 'error on fetching user {} {}'.format(username, e))
            continue

        try:
            targets = instagram.followers(username_id)
            if targets['status'] != 'ok':
                log_event(
                    'error',
                    'invalid Instagram status ' + followers['status'],
                )
                continue
        except Exception as e:
            log_event('warn', 'error on fetching followers {}'.format(e))
            continue

        for target in tqdm(shuffle_list(targets['users'], limit=limit_users)):
            username = target['username']

            try:
                image = url_to_image(target['profile_pic_url'])
                gender = detector.process(image)
                graph.add_edge(username, gender,
                               follower['username'], follower['gender'])
            except Exception as e:
                log_event(
                    'warn', 'error on fetching user {} {}'.format(username, e))
                continue

    log_event('info', 'logging out from Instagram ...')
    instagram.logout()
