import os
import sys
import time
import workerpool
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


class ProcessJob(workerpool.Job):
    def __init__(self, graph, detector, target, follower):
        self.graph = graph
        self.detector = detector
        self.target = target
        self.follower = follower

    def run(self):        
        username = self.target['username']
        try:
            image = url_to_image(self.target['profile_pic_url'])
            gender = self.detector.process(image)
            self.graph.add_edge(
                username,
                gender,
                self.follower['username'],
                self.follower['gender'],
            )
            return 1
        except Exception as e:
            log_event(
                'warn',
                'error on fetching user {} {}'.format(username, e),
            )
        return 0


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
    current_gender = 'unknown'

    log_event('info', 'trying to connect to Instagram ...')
    if not instagram.login():
        log_event('error', "couldn't connect to Instagram !")
        sys.exit(1)

    log_event('info', 'fetching logged-in user followers ...')
    followers = instagram.followers(instagram.username_id)
    if followers['status'] != 'ok':
        log_event('error', 'invalid Instagram status ' + followers['status'])
        sys.exit(2)

    for follower in tqdm(shuffle_list(followers['users'], limit=limit_users)):
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

        targets = shuffle_list(targets['users'], limit=limit_users)
        map_list = [(graph, detector, target, follower) for target in targets]

        pool = workerpool.WorkerPool(size=8)
        
        for target in targets:
            job = ProcessJob(graph, detector, target, follower)
            pool.put(job)

        pool.shutdown()
        pool.wait()

    log_event('info', 'logging out from Instagram ...')
    instagram.logout()
