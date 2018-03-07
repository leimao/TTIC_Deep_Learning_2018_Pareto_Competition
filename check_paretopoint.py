# coding: utf-8

from __future__ import print_function
import argparse
import time
import math
import matplotlib
import matplotlib.pyplot as plt
import mimetypes
import os
import sys
from functools import partial
from uuid import uuid4
from tornado import gen, httpclient, ioloop
from tornado.options import define, options
import json


parser = argparse.ArgumentParser(description='Submit your model/check the pareto point through http protocol')
                    
# three components must submit for final grading
'''                    
parser.add_argument('--model', type=str, default='./checkpoint/model.pt',
                    help='your trained model')
'''
parser.add_argument('--model', type=str, default='model.pt',
                    help='your trained model')
parser.add_argument('--model_module', type=str, default='model.py',
                    help='your python file "model.py" ')
parser.add_argument('--main_module', type=str, default='main.py',
                    help='your python file "main.py"')

# fake identity for pareto point display only
parser.add_argument('--pseudonym', type=str, default='Depth_Charge_New',
                    help='pseudonym for display purpose')

# real identity information
parser.add_argument('--name', type=str, default='Lei_Mao',
                    help='your real name')
parser.add_argument('--student_id', type=str, default='XXXXXXXX',
                    help='your student id')

args = parser.parse_args()
plt.switch_backend('agg')

########################################################### Http submision utils ###################################
@gen.coroutine
def multipart_producer(boundary, filenames, write):
    boundary_bytes = boundary.encode()

    for filename in filenames:
        filename_bytes = filename.encode()
        mtype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        buf = (
            (b'--%s\r\n' % boundary_bytes) +
            (b'Content-Disposition: form-data; name="%s"; filename="%s"\r\n' %
             (filename_bytes, filename_bytes)) +
            (b'Content-Type: %s\r\n' % mtype.encode()) +
            b'\r\n'
        )
        yield write(buf)
        with open(filename, 'rb') as f:
            while True:
                # 16k at a time.
                chunk = f.read(16 * 1024)
                if not chunk:
                    break
                yield write(chunk)

        yield write(b'\r\n')

    yield write(b'--%s--\r\n' % (boundary_bytes,))

@gen.coroutine
def post(filenames, pseudonym, name, student_id):

    client = httpclient.AsyncHTTPClient()
    httpclient.AsyncHTTPClient.configure(None, defaults=dict(connect_timeout=2000, request_timeout=3000))
    boundary = uuid4().hex
    headers = {'Content-Type': 'multipart/form-data; boundary=%s' % boundary}
    producer = partial(multipart_producer, boundary, filenames)
    print ('http://128.135.8.238:5000/upload?pseudonym=%s&name=%s&student_id=%s'% (pseudonym, name, student_id))

    response = yield client.fetch('http://128.135.8.238:5000/upload?pseudonym=%s&name=%s&student_id=%s'% (pseudonym, name, student_id),
                                  method='POST',
                                  headers=headers,
                                  body_producer=producer)

    print(response)

def post_status(pseudonym, ratio, perp):

    client = httpclient.HTTPClient()
    response = client.fetch('http://128.135.8.238:5000/update_status?pseudonym=%s&ratio=%f&perp=%f' % (pseudonym, ratio, perp)
                )            
    print(response)


########################################################### Display & Submission Function ###################################

def submit_final_model():

    # submit the identity information, remove the space
    pseudonym = args.pseudonym
    name = args.name
    student_id = args.student_id
    
    # submit the file
    main_module = args.main_module
    model_module = args.model_module
    model = args.model
    submissions = [main_module, model_module, model]
    ioloop.IOLoop.current().run_sync(lambda: post(submissions, pseudonym, name, student_id))


def submit_current_status(ratio, perp):

    # submit the identity information
    pseudonym = args.pseudonym
    post_status(pseudonym, ratio, perp)

def fetch_current_status():

    
    http_client = httpclient.HTTPClient()
    try:
        response = http_client.fetch('http://128.135.8.238:5000/Paretopoint')
        print(response.body)
    except httpclient.HTTPError as e:
        print("Error:", e)
    http_client.close()

    # write it as dict
    with open('paretopoint.json', 'wb+') as f:
        f.write(response.body)

    # read current status and display it to the paretopoint.png
    fp = open('paretopoint.json', 'r')
    status = json.load(fp)
    fp.close()
    plt.ticklabel_format(useOffset=False)
    plt.axis([0, 1, 0, 150])
    all_points_x = []
    all_points_y = []
    labels = []
    for key in status.keys():
        labels.append(key)
        all_points_y.append(status[key][0])
        all_points_x.append(status[key][1])

    fig, ax = plt.subplots()
    ax.scatter(all_points_x, all_points_y)
    ax.set_xlabel('Ratio: Training Time/Baseline Time')
    ax.set_ylabel('Perplexity')
    for i, label in enumerate(labels):
        ax.annotate(label, (all_points_x[i], all_points_y[i]))

    plt.savefig("paretopoint.png")

def main():

    print("Use this code to submit your model or check the current paretopoint")
    print("Options: \n")
    print("0: submit your final model and code\n")
    print("1: check current paretopoint and display it\n")
    print("2: submit your current status\n")
    option = input("Type your choice: ")

    if int(option) == 0:
        submit_final_model()
    elif int(option) == 1:
        fetch_current_status()        
    else:
        ratio = input("Type your run time ratio: ")
        perp = input("Type your valid perplexity: ")
        submit_current_status(float(ratio), float(perp))

if __name__== "__main__":
  main()