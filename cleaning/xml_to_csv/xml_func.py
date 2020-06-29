import xml.etree.cElementTree as ET
import pandas as pd
import time


def parse_xml(file):

    #sample_xml_file = '/Users/ramonperez/OneDrive/CoderAcademy/data_sessions/apple_health_export.xml'

    root = ET.parse(str(file)).getroot()

    start_time = time.time()
    counter = 0
    jobs_list = []
    for node in root:
        job_dict = {}
        for elem in node:
            # print(elem.tag)
            # print(elem.text)
            if elem.tag not in job_dict.keys():
                if elem.tag == "CanonSkills":
                    skill_dict = {}
                    for skill in elem:
                        if skill.attrib:
                            skill_dict.update({skill.attrib['name']: skill.attrib['clusterName']})

                    job_dict[elem.tag] = skill_dict

                else:
                    job_dict[elem.tag] = elem.text
            else:
                print(elem.tag)
        jobs_list.append(job_dict)
        counter += 1
        if counter % 1000 == 0:
            elapsed_time = time.time() - start_time
            print(str(counter) + " records done, elapsed time: "+str(elapsed_time)+" sec")

    out_df = pd.DataFrame(jobs_list)

    out_df.to_csv(
        '/Users/ramonperez/OneDrive/CoderAcademy/data_sessions/apple_health.csv', index=False)
