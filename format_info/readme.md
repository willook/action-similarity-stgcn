- distributions
    # Florence: (23, 32, 15, 3) max(991.4373503654234, 924.6611328125, 3553.3125), min(-894.992427702873, -1635.0623779296875, 2160.5501275831652)
    # NTU: max(3.209906 1.728893 5.110941), min(-2.834219 -2.15311 0.0)
    # bpe keypoints_extractor: x_max=1682.09619140625, x_min=127.22637939453125, y_max=1015.7581176757812, y_min=4.100187301635742


- stgcn 모델 입력: (batch_size, seq_len(32), keypoints(15), coordinate(2 or 3))
e.g. (1, 32, 15, 2)

- Florence dataset (stgcn 입력)
torch.Size([1, 32, 15, 3])
0 Head: f1-f3
1 Neck: f4-f6
2 Spine: f7-f9
3 Left Shoulder: f10-f12
4 Left Elbow: f13-f15
5 Left Wrist: f16-f18
6 Right Shoulder: f19-f21
7 Right Elbow: f22-f24
8 Right Wrist: f25-f27
9 Left Hip: f28-f30
10 Left Knee: f31-f33
11 Left Ankle: f34-f36
12 Right Hip: f37-f39
13 Right Knee: f40-f42
14 Right Ankle: f43-f45

- ntu rgbd dataset
0-base of the spine 
1-middle of the spine -> 3
2-neck -> 2
3-head -> 1
4-left shoulder -> 4 
5-left elbow -> 5
6-left wrist -> 6
7-left hand 
8-right shoulder 
9-right elbow 
10-right wrist 
11-right hand 
12-left hip 
13-left knee 
14-left ankle 
15-left foot 
16-right hip 
17-right knee 
18-right ankle 
19-right foot 
20-spine 
21-tip of the left hand 
22-left thumb 
23-tip of the right hand 
24-right thumb

- BPE keypoint extract
nose = np.array([keypoints['nose']['x'], keypoints['nose']['y']])
right_shoulder = np.array([keypoints['right_shoulder']['x'], keypoints['right_shoulder']['y']])
right_elbow = np.array([keypoints['right_elbow']['x'], keypoints['right_elbow']['y']])
right_wrist = np.array([keypoints['right_wrist']['x'], keypoints['right_wrist']['y']])
left_shoulder = np.array([keypoints['left_shoulder']['x'], keypoints['left_shoulder']['y']])
left_elbow = np.array([keypoints['left_elbow']['x'], keypoints['left_elbow']['y']])
left_wrist = np.array([keypoints['left_wrist']['x'], keypoints['left_wrist']['y']])
right_hip = np.array([keypoints['right_hip']['x'], keypoints['right_hip']['y']])
right_knee = np.array([keypoints['right_knee']['x'], keypoints['right_knee']['y']])
right_ankle = np.array([keypoints['right_ankle']['x'], keypoints['right_ankle']['y']])
left_hip = np.array([keypoints['left_hip']['x'], keypoints['left_hip']['y']])
left_knee = np.array([keypoints['left_knee']['x'], keypoints['left_knee']['y']])
left_ankle = np.array([keypoints['left_ankle']['x'], keypoints['left_ankle']['y']])
neck = (right_shoulder + left_shoulder) / 2
mid_hip = (right_hip + left_hip) / 2

# returns of BPE keypoint extractor
 '8': [{'frame': 6,
        'keypoints': {'box': {'x1': 293.32562255859375,
                              'x2': 317.6053161621094,
                              'y1': 137.32386779785156,
                              'y2': 237.1590576171875},
                      'left_ankle': {'score': 0.026342138648033142,
                                     'x': 317.4455871582031,
                                     'y': 236.99905395507812},
                      'left_ear': {'score': 0.06924092024564743,
                                   'x': 317.4455871582031,
                                   'y': 165.6425018310547},
                      'left_elbow': {'score': 0.02880089543759823,
                                     'x': 317.4455871582031,
                                     'y': 236.99905395507812},
                      'left_eye': {'score': 0.186357319355011,
                                   'x': 308.81988525390625,
                                   'y': 160.5227508544922},
                      'left_hip': {'score': 0.009931959211826324,
                                   'x': 313.29248046875,
                                   'y': 157.96286010742188},
                      'left_knee': {'score': 0.033073555678129196,
                                    'x': 311.0561828613281,
                                    'y': 174.9220428466797},
                      'left_shoulder': {'score': 0.005995157640427351,
                                        'x': 294.1242980957031,
                                        'y': 217.79998779296875},
                      'left_wrist': {'score': 0.016560452058911324,
                                     'x': 317.4455871582031,
                                     'y': 236.99905395507812},
                      'nose': {'score': 0.1180301159620285,
                               'x': 315.20928955078125,
                               'y': 167.5624237060547},
                      'right_ankle': {'score': 0.030274515971541405,
                                      'x': 312.33404541015625,
                                      'y': 236.99905395507812},
                      'right_ear': {'score': 0.06288894265890121,
                                    'x': 296.99951171875,
                                    'y': 171.08224487304688},
                      'right_elbow': {'score': 0.009037970565259457,
                                      'x': 311.0561828613281,
                                      'y': 236.99905395507812},
                      'right_eye': {'score': 0.22574958205223083,
                                    'x': 309.1393737792969,
                                    'y': 160.5227508544922},
                      'right_hip': {'score': 0.009368234314024448,
                                    'x': 293.4853515625,
                                    'y': 236.99905395507812},
                      'right_knee': {'score': 0.016310380771756172,
                                     'x': 293.4853515625,
                                     'y': 236.99905395507812},
                      'right_shoulder': {'score': 0.014638766646385193,
                                         'x': 294.76324462890625,
                                         'y': 203.4006805419922},
                      'right_wrist': {'score': 0.008203564211726189,
                                      'x': 317.4455871582031,
                                      'y': 236.99905395507812},
                      'score': 0.6168033480644226,
                      'track_id': '8'}},
       {'frame': 7,
        'keypoints': {'box': {'x1': 286.7440490722656,
                              'x2': 316.2117614746094,
                              'y1': 137.6715850830078,
                              'y2': 213.58114624023438},
                      'left_ankle': {'score': 0.00902376975864172,
                                     'x': 316.0533142089844,
                                     'y': 137.8310546875},
                      'left_ear': {'score': 0.043469689786434174,
                                   'x': 316.0533142089844,
                                   'y': 161.7521514892578},
                      'left_elbow': {'score': 0.013617690652608871,
                                     'x': 316.0533142089844,
                                     'y': 213.4216766357422},
                      'left_eye': {'score': 0.19694043695926666,
                                   'x': 316.0533142089844,
                                   'y': 159.2005615234375},
                      'left_hip': {'score': 0.006103092804551125,
                                   'x': 307.18133544921875,
                                   'y': 156.64898681640625},
                      'left_knee': {'score': 0.013192633166909218,
                                    'x': 300.84417724609375,
                                    'y': 174.1911163330078},
                      'left_shoulder': {'score': 0.013777635991573334,
                                        'x': 311.6173095703125,
                                        'y': 213.4216766357422},
                      'left_wrist': {'score': 0.054869379848241806,
                                     'x': 311.9341735839844,
                                     'y': 141.65843200683594},
                      'nose': {'score': 0.14583078026771545,
                               'x': 311.6173095703125,
                               'y': 164.94161987304688},
                      'right_ankle': {'score': 0.011104671284556389,
                                      'x': 316.0533142089844,
                                      'y': 137.8310546875},
                      'right_ear': {'score': 0.07267913222312927,
                                    'x': 286.9024963378906,
                                    'y': 166.21742248535156},
                      'right_elbow': {'score': 0.03321199119091034,
                                      'x': 286.9024963378906,
                                      'y': 213.4216766357422},
                      'right_eye': {'score': 0.22231654822826385,
                                    'x': 301.4779052734375,
                                    'y': 157.92477416992188},
                      'right_hip': {'score': 0.005575106479227543,
                                    'x': 286.9024963378906,
                                    'y': 213.4216766357422},
                      'right_knee': {'score': 0.02075568586587906,
                                     'x': 286.9024963378906,
                                     'y': 213.4216766357422},
                      'right_shoulder': {'score': 0.013077475130558014,
                                         'x': 286.9024963378906,
                                         'y': 194.28480529785156},
                      'right_wrist': {'score': 0.04317611828446388,
                                      'x': 310.9836120605469,
                                      'y': 141.65843200683594},
                      'score': 0.7304453253746033,
                      'track_id': '8'}}],