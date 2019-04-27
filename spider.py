import os
import sys
import time
from graph import Graph
from imutils import url_to_image
from os.path import join as join_path
from instagram import Instagram
from faceutil import Detector


def log_event(status, text):
    unixtime = int(time.time())
    print "[{} {:>5}] {}".format(unixtime, status.upper(), text)


if __name__ == "__main__":
    detector = Detector(
        join_path('models', 'shape_predictor_68_face_landmarks.dat'),
        join_path('models', 'dlib_face_recognition_resnet_model_v1.dat'),
        join_path('models', 'gender.pickle'),
    )
    graph = Graph('')
    instagram = Instagram(
        os.getenv('INSTAGRAM_USERNAME'),
        os.getenv('INSTAGRAM_PASSWORD'),
    )
    current_user = os.getenv('INSTAGRAM_USERNAME')

    log_event('info', 'trying to connect to Instagram ...')
    if not instagram.login():
        log_event('error', "couldn't connect to Instagram !")
        sys.exit(1)

    log_event('info', 'fetching logged-in user followers ...')
    followers = instagram.followers(instagram.username_id)
    if followers['status'] != 'ok':
        log_event('error', 'invalid Instagram status ' + followers['status'])
        sys.exit(2)
    
    for follower in followers['users'][:10]:
        username = follower['username']
        username_id = follower['pk']

        image = url_to_image(follower['profile_pic_url'])
        gender = detector.process(image)
        graph.add_node(username, gender)
        graph.add_edge(current_user, username)

        targets = instagram.followers(username_id)
        if targets['status'] != 'ok':
            log_event(
                'error',
                'invalid Instagram status ' + followers['status'],
            )
            continue

        for target in targets:  
            username = target['username']
            image = url_to_image(target['profile_pic_url'])
            gender = detector.process(image)

            graph.add_node(username, gender)
            graph.add_edge(follower['username'], username)