from zeep import Client
import pandas as pd
from time import time, sleep

client = Client('https://colondexsrv.its.nv.gov/tmddws/TmddWS.svc?singleWsdl')
print(client.get_type('ns0:DeviceInformationRequest'))

# IDENTIFICATION
user = "UNLV_TRC_RTIS"
password = "+r@^~Tr&lt;R?|$"
organization = "unlv.edu"
center = "UNLV_TRC"
# CENTER REQUESTED
center_req = "FAST"
device_type = "detector"
device_info_data = "device data"
device_info_inventory = "device inventory"
organisation_req = "its.nv.gov"

auth = {
    "user-id": user,
    "password": password
}
org_info = {
    "organization-id": organization,
    "center-contact-list": {
        "center-contact-details": {
            "center-id": center
        }
    }
}
org_info_req = {
    "organization-id": organisation_req,
    "center-contact-list": {
        "center-contact-details": {
            "center-id": center_req
        }
    }
}

detectors_nb_list = [
    '440_1_335',
    '439_1_334',
    '439_2_333',
    '439_3_332',
    '438_1_331',
    '438_2_330',
    '438_3_329',
    '359_1_325',
    '358_1_325',
    '358_2_320',
    '358_3_319',
    '357_1_312',
    '357_2_311',
    '357_3_310',
    '356_1_309',
    '356_2_308',
    '355_1_156',
    '355_2_153',
    '355_3_155',
    '354_1_79',
    '354_2_144',
    '354_3_145',
    '32_1_142',
    '34_1_94',
    '39_2_88',
    '48_2_83',
    '49_1_82',
    '49_2_12',
    '49_3_15',
    '58_2_17',
    '59_1_18',
    '59_2_18',
    '70_2_21',
    '71_2_23',
    '72_1_22',
    '72_2_28',
    '89_1_28',
    '89_2_30',
    '97_1_33',
    '97_2_33',
    '97_3_38',
    '99_1_35',
    '110_1_41',
    '112_2_44',
    '113_2_45',
    '122_2_48',
    '124_2_49',
    '137_1_80',
    '138_1_53',
    '138_2_55',
    '146_2_238',
    '148_2_58',
    '149_2_240',
    '160_2_242',
    '396_1_243',
    '396_2_246',
    '396_3_246',
    '397_1_247',
    '397_2_248',
    '398_1_249',
    '398_2_251']

# %%  Requesting inventory
result = client.service.dlDetectorInventoryRequest(auth, org_info, org_info_req, device_type, device_info_inventory)
detect_list_inv = []
for detect in result:
    detect_list_inv.append(detect['detector-station-inventory-header']['device-id'])

detect_list_inv_filt = []
for detect_id in detect_list_inv:
    if detect_id not in detect_list_inv_filt:
        detect_list_inv_filt.append(detect_id)

# print(detect_list_inv_filt)
print(len(detect_list_inv_filt))

# %% Requesting data

parameters_data = {
    "authentication": auth,
    "organization-information": org_info,
    "organization-requesting": org_info_req,
    "device-type": device_type,
    "device-information-type": device_info_data
}

first_time = True
columns = ['DateTimeStamp', 'DetectorID', 'Volume', 'Occupancy', 'Speed']
data_table_all = pd.DataFrame(columns=columns)
while True:
    sleep(60 - time() % 60)
    result = client.service.dlDetectorDataRequest(parameters_data)
    detectorDataItem = result[0]
    detectorList = 'detector-list'
    detectorList = detectorDataItem[detectorList]
    detectorDataDetail = 'detector-data-detail'
    detectorDataDetail = detectorList[detectorDataDetail]
    nbDetector = len(detectorDataDetail)

    #if first_time:
    detect_list_data = []
    for detection in detectorDataDetail:
        detect_list_data.append(detection['station-id'])

    detect_list_data_filt = []
    for detect_id in detect_list_data:
        if detect_id not in detect_list_data_filt:
            detect_list_data_filt.append(detect_id)

    #print(detect_list_data_filt)
    print(f'Number of detectors = {len(detect_list_data_filt)}')
    #first_time = False

    # Print current date
    currentDetector = detectorDataDetail[0]
    detectionTimeStamp = 'detection-time-stamp'
    detectionTimeStamp = currentDetector[detectionTimeStamp]
    date = 'date'
    hour = 'time'
    date = detectionTimeStamp[date]
    hour = detectionTimeStamp[hour]
    dateTime = f'{date[0:4]}-{date[4:6]}-{date[6:]} {hour[0:2]}:{hour[2:4]}:{hour[4:6]}'
    print(dateTime)

    previous_detect_id = 'init'
    volume_sum = 0
    occupancy_sum = 0
    speed_sum = 0
    lane_count = 0
    data_count = 0
    for detection in detectorDataDetail:
        if detection['station-id'] in detectors_nb_list:
            detect_id = detection['station-id']
            if detect_id != previous_detect_id:
                if previous_detect_id == 'init' or detection == detectorDataDetail[-1]:
                    data_table = pd.DataFrame(columns=columns)
                else:
                    data_table = data_table.append(pd.DataFrame([[dateTime, previous_detect_id,
                                                                  volume_sum / lane_count, occupancy_sum / lane_count,
                                                                  speed_sum / lane_count]],
                                                                columns=columns))
                    data_count += 1
                previous_detect_id = detect_id
                lane_count = 1
                volume_sum = detection['vehicle-count']
                occupancy_sum = detection['vehicle-occupancy']
                speed_sum = detection['vehicle-speed']
            else:
                lane_count += 1
                volume_sum += detection['vehicle-count']
                occupancy_sum += detection['vehicle-occupancy']
                speed_sum += detection['vehicle-speed']

    # Last data of the table
    data_table = data_table.append(pd.DataFrame([[dateTime, detect_id,
                                                  volume_sum / lane_count, occupancy_sum / lane_count,
                                                  speed_sum / lane_count]],
                                                columns=columns))
    data_count += 1

    data_table_all = data_table_all.append(data_table)
    print(f'Number of NB detectors = {data_count}')
    print(data_table_all.tail(data_count))
