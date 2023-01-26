# -*- coding: utf-8 -*-
'''
Reproject images from equirectangular to cubemap.
- Credits to Tim Alpherts for the base script.
- Partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet.

@author: Andrea Lombardo
'''

import os
import sys

import multiprocessing
import time
import argparse

import numpy as np
from PIL import Image
import lib.vrProjector as vrProjector
from tqdm import tqdm                

class Consumer(multiprocessing.Process):

    def __init__(self, task_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            next_task()
            self.task_queue.task_done()
        return

def split(args, img_path):

    img = os.path.join(args.input_dir, img_path)
    # Get the pano id by removing .jpg from img_path
    pano_id = img_path[:-4]

    # VrProjector the images 
    size = args.size
    eq = vrProjector.EquirectangularProjection()
    eq.loadImage(img)
    cb = vrProjector.CubemapProjection()
    cb.initImages(size,size)
    cb.reprojectToThis(eq)

    # retrieve front back left right
    front = Image.fromarray(np.uint8(cb.front))
    right = Image.fromarray(np.uint8(cb.right))
    back = Image.fromarray(np.uint8(cb.back))
    left = Image.fromarray(np.uint8(cb.left))

    # make directory, with panoid as name, to save them in
    directory = os.path.join(args.output_dir, pano_id)
    if not os.path.exists(directory):
            os.makedirs(directory)
    # save them in that dir
    front.save(os.path.join(directory, 'front.jpg'))
    right.save(os.path.join(directory, 'right.jpg'))
    back.save(os.path.join(directory, 'back.jpg'))
    left.save(os.path.join(directory, 'left.jpg'))

    print('saved {}!'.format(pano_id))

class Task(object):
    def __init__(self, args, img_path):
        #print('Task init')
        self.img_path = img_path
        self.args = args
        
    def __call__(self):
        #print('Task call')
        split(self.args, self.img_path)
        
        #seq_24(self.args, self.info)
        #print('Inside call', self.info)
        
    def __str__(self):
        return self.info['filename']


def main(args):
    
    directory = os.path.join(os.getcwd(),args.output_dir)
    if not os.path.exists(directory):
                os.makedirs(directory)
                
    output_dir = args.output_dir
    
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()

    # Start consumers
    #num_consumers = multiprocessing.cpu_count() * 2
    #num_consumers = multiprocessing.cpu_count()
    num_consumers = 1
    print('Creating %d consumers' % num_consumers)
    consumers = [Consumer(tasks) for i in range(num_consumers)]
    for w in consumers:
        w.start()

    # Give the list of images to the Taskforce
    img_list = os.listdir(args.input_dir)
    
    print('Init zip..')

    for img_path in img_list:
        print('Adding task for {}'.format(img_path))
        tasks.put(Task(args, img_path))

    # Add a poison pill for each consumer
    for i in range(num_consumers):
        tasks.put(None)

    pbar = tqdm(total=tasks.qsize())

    last_queue = tasks.qsize()

    while tasks.qsize() > 0:
        diff = last_queue - tasks.qsize()
        pbar.update(diff)
        last_queue = tasks.qsize()
        time.sleep(0.2)

    # Wait for all of the tasks to finish
    tasks.join()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_dir', type=str, default = 'res/dataset/reoriented') 
    parser.add_argument('--output_dir', type=str, default = 'res/dataset/reprojected')
    parser.add_argument('--size', type=int, default = 512)
    
    args = parser.parse_args()
    
    main(args)