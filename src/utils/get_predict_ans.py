import xml.dom.minidom
import zipfile

import pandas as pd
import requests
from pip._vendor.urllib3 import encode_multipart_formdata

from path_manager.path_manager import DataPathManager


def get_dict(**dict):
    return dict


def get_predict_auc(data_name=None, predict_ans=None, model_idx=None, output_path_manager=None):
    # zip files to output.zip
    data_path_manager = DataPathManager()
    zip_name = data_path_manager.get_output_zip_path()
    files = [data_path_manager.get_output_csv_path("A"), data_path_manager.get_output_csv_path("B")]
    # store predict ans
    if data_name is not None:
        if model_idx is not None:
            pd.Series(predict_ans).to_csv(output_path_manager.get_output_csv_path(data_name), header=None,
                                          index=None)
            if data_name == "A":
                files[0] = output_path_manager.get_output_csv_path(data_name)
            else:
                files[1] = output_path_manager.get_output_csv_path(data_name)
            zip_name = output_path_manager.get_output_zip_path()
        else:
            pd.Series(predict_ans).to_csv(data_path_manager.get_output_csv_path(data_name), header=None, index=None)

    zp = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for file in files:
        zp.write(file)
    zp.close()

    # get ans
    url = 'http://eval-env.eba-5u39qmpg.us-west-2.elasticbeanstalk.com/submit'
    filename = "output.zip"
    data = {'result': (filename, open(zip_name, 'rb').read())}
    encode_data = encode_multipart_formdata(data)
    data = encode_data[0]
    header = {'Content-Type': encode_data[1]}
    r = requests.post(url, headers=header, data=data)
    # parser
    dom = xml.dom.minidom.parseString(r.text)
    root = dom.documentElement
    nodes = root.getElementsByTagName('p')

    s = nodes[0].childNodes[0].data
    auc_A = float(s[s.find(':') + 2:-1])
    s = nodes[1].childNodes[0].data
    auc_B = float(s[s.find(':') + 2:-1])
    s = nodes[2].childNodes[0].data
    result = float(s[s.find(':') + 2:-1])
    return get_dict(auc_A=auc_A, auc_B=auc_B, result=result)
