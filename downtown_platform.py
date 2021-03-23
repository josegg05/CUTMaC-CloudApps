import paho.mqtt.client as mqtt
import sys
import time
import json
from os import system, name
import numpy as np



def subscribe_platform_topics(client, platform_topics):
    for topic in platform_topics:
        client.subscribe(topic)
        print(f'Subscribed to: {topic}')
    time.sleep(len(platform_topics))
    return


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    # Define MQTT topics
    print(platform_topics_selected)
    subscribe_platform_topics(client, platform_topics_selected)
    print('*********** Start receiving messages ***********\n\n')


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global id_list
    msg.payload = msg.payload.decode("utf-8")
    #clear()
    #print(f'Topic: {msg.topic}')
    print(f'Message: {msg.payload}')
    json_object = json.loads(msg.payload)
    json_formatted_str = json.dumps(json_object, indent=2)
    #print(f'Message:{json_formatted_str}')
    if json_object['lidarObject']['objectClass'] in ['VEHICLE', 'HUMAN', 'UNIDENTIFIED']:
        #print(f'Message:{json_formatted_str}')
        print(f'Intersection ID: {json_object["intersectionId"]} - Class: {json_object["lidarObject"]["objectClass"]} - Speed:{json_object["lidarObject"]["velocity"]["speed"]}')
        id_list.append(json_object["intersectionId"])

        if len(id_list) >= 5000:
            print(np.unique(np.array(id_list)))
            np.savetxt('id_list_test.txt', np.unique(np.array(id_list)), fmt='%s')
            sys.exit()




# clear screen function
def clear():
    #windows
    if name == 'nt':
        _ = system('cls')
    #nix
    else:
        _ = system('clear')

def mqtt_conf(mqtt_broker_ip) -> mqtt.Client:
    broker_address = mqtt_broker_ip
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_address)  # connect to broker
    return client


def run():

    return


# if you have global variables, you must
# initialize them outside the main block
# initialize()
if __name__ == '__main__':
    id_list = []
    option = input('Select an Option:\n 1. Select a topic from the topic list \n 2. Enter your own topic\n')

    if option == '1' or option == '2':
        if option == '1':
            platform_topics = ['cisco/edge-intelligence/telemetry/clv/streaming/rawSpat',
                               'cisco/edge-intelligence/telemetry/clv/streaming/j2735Spat',
                               'cisco/edge-intelligence/telemetry/clv/streaming/lidarObjects',
                               'cisco/edge-intelligence/telemetry/clv/streaming/j2735Psm',
                               'cisco/edge-intelligence/telemetry/clv/streaming/j2735Bsm',
                               'cisco/edge-intelligence/telemetry/clv/streaming/bsmTopic',
                               'bsmTopic',
                               'streaming/bsmTopic']

            topic_idx = input(f'Select one (Ex: 4) or many (Ex: 1,2,5) topics:\n'
                              f'0. cisco/edge-intelligence/telemetry/clv/streaming/rawSpat\n'
                              f'1. cisco/edge-intelligence/telemetry/clv/streaming/j2735Spat\n'
                              f'2. cisco/edge-intelligence/telemetry/clv/streaming/lidarObjects\n'
                              f'3. cisco/edge-intelligence/telemetry/clv/streaming/j2735Psm\n'
                              f'4. cisco/edge-intelligence/telemetry/clv/streaming/j2735Bsm\n'
                              f'5. cisco/edge-intelligence/telemetry/clv/streaming/bsmTopic\n'
                              f'6. bsmTopic\n'
                              f'7. streaming/bsmTopic\n'
                              f'8. all topics\n')

            topic_idx = list(topic_idx.split(","))
            topic_idx = list(map(int, topic_idx))
            print(topic_idx)

            if len(topic_idx) == 1:
                if topic_idx[0] == 8:
                    platform_topics_selected = platform_topics
                elif topic_idx[0] <= len(platform_topics):
                    platform_topics_selected = [platform_topics[topic_idx[0]]]
                else:
                    print('Invalid Option. Program Finished')
                    sys.exit()
            else:
                platform_topics_selected = []
                for top_idx in topic_idx:
                    if top_idx <= len(platform_topics):
                        platform_topics_selected.append(platform_topics[top_idx])

        elif option == '2':
            topic = input("Enter your topic:\n")
            platform_topics_selected = [topic]

        mqtt_broker_ip = '10.30.4.149'
        # Start mqtt connection
        client = mqtt_conf(mqtt_broker_ip)
        # client.loop_start()  # Necessary to maintain connection

        client.loop_forever()
        # Reset Loop
        # while True:
        #     msg_dic = []
        #     run()

    else:
        print('Invalid Option. Program Finished')
        sys.exit()

