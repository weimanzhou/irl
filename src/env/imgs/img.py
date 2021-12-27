import numpy as np

car = np.array([[[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [212, 212, 209],
                 [209, 209, 207],
                 [209, 209, 206],
                 [207, 207, 205],
                 [211, 211, 208],
                 [211, 211, 208],
                 [211, 211, 208],
                 [211, 211, 208],
                 [209, 209, 207],
                 [209, 209, 206],
                 [207, 207, 205],
                 [211, 211, 208],
                 [211, 211, 211],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [163, 163, 163],
                 [165, 165, 165],
                 [165, 165, 165],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [163, 163, 163],
                 [165, 165, 165],
                 [165, 165, 165],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [38, 174, 255],
                 [6, 165, 255],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[80, 80, 255],
                 [76, 76, 252],
                 [77, 77, 250],
                 [79, 79, 249],
                 [80, 80, 246],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [77, 77, 250],
                 [79, 79, 249],
                 [80, 80, 246],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [99, 102, 251],
                 [109, 112, 251],
                 [76, 76, 252],
                 [75, 75, 251],
                 [0, 0, 0]],
                [[76, 76, 251],
                 [106, 104, 235],
                 [198, 187, 186],
                 [168, 160, 203],
                 [76, 76, 252],
                 [101, 101, 244],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [101, 101, 244],
                 [76, 76, 252],
                 [168, 160, 203],
                 [198, 187, 186],
                 [96, 95, 241],
                 [168, 160, 203],
                 [198, 187, 186],
                 [201, 191, 188],
                 [97, 96, 239],
                 [0, 0, 0]],
                [[76, 76, 251],
                 [117, 114, 230],
                 [239, 225, 165],
                 [199, 188, 187],
                 [76, 76, 252],
                 [179, 182, 219],
                 [106, 134, 241],
                 [106, 134, 241],
                 [106, 134, 241],
                 [106, 134, 241],
                 [182, 186, 218],
                 [76, 76, 252],
                 [199, 188, 187],
                 [239, 225, 165],
                 [117, 114, 230],
                 [199, 188, 187],
                 [239, 225, 165],
                 [243, 232, 180],
                 [125, 121, 223],
                 [0, 0, 0]],
                [[76, 76, 251],
                 [96, 95, 241],
                 [158, 151, 209],
                 [137, 132, 219],
                 [76, 76, 252],
                 [101, 101, 244],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [101, 101, 244],
                 [76, 76, 252],
                 [137, 132, 219],
                 [158, 151, 209],
                 [86, 85, 246],
                 [137, 132, 219],
                 [158, 151, 209],
                 [164, 157, 209],
                 [87, 86, 244],
                 [0, 0, 0]],
                [[53, 118, 252],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [40, 150, 254],
                 [42, 147, 254],
                 [0, 0, 0]],
                [[110, 110, 239],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [131, 144, 234],
                 [130, 142, 234],
                 [0, 0, 0]],
                [[76, 76, 251],
                 [76, 76, 252],
                 [76, 76, 252],
                 [75, 75, 252],
                 [73, 66, 119],
                 [72, 64, 90],
                 [73, 70, 192],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [75, 75, 252],
                 [72, 66, 141],
                 [72, 64, 90],
                 [72, 69, 186],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [0, 0, 0]],
                [[76, 76, 251],
                 [76, 76, 252],
                 [76, 76, 252],
                 [72, 64, 82],
                 [139, 133, 127],
                 [167, 164, 162],
                 [102, 95, 85],
                 [74, 71, 217],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [76, 76, 252],
                 [72, 66, 141],
                 [111, 103, 95],
                 [167, 164, 162],
                 [101, 93, 84],
                 [74, 71, 205],
                 [76, 76, 252],
                 [76, 76, 252],
                 [0, 0, 0]],
                [[80, 80, 255],
                 [76, 76, 251],
                 [76, 76, 251],
                 [72, 61, 50],
                 [184, 183, 185],
                 [118, 123, 126],
                 [135, 130, 124],
                 [70, 61, 104],
                 [76, 76, 251],
                 [76, 76, 251],
                 [76, 76, 251],
                 [76, 76, 251],
                 [72, 62, 50],
                 [168, 165, 163],
                 [99, 104, 108],
                 [177, 174, 172],
                 [71, 62, 62],
                 [76, 76, 251],
                 [76, 76, 251],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [72, 62, 50],
                 [128, 124, 119],
                 [168, 165, 163],
                 [102, 94, 86],
                 [64, 64, 48],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [70, 61, 51],
                 [123, 117, 110],
                 [176, 173, 172],
                 [123, 117, 110],
                 [72, 60, 48],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [72, 61, 50],
                 [72, 62, 50],
                 [72, 56, 48],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [70, 61, 51],
                 [72, 62, 50],
                 [72, 60, 48],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]])

target = np.array([[[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 21, 213],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 30, 215],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 29, 215],
                    [5, 30, 217],
                    [6, 29, 216],
                    [5, 30, 216],
                    [5, 31, 214],
                    [7, 31, 217],
                    [7, 31, 216],
                    [6, 30, 216],
                    [5, 30, 217],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 30, 215],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 30, 215],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 30, 215],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 29, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 31, 215],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 29, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 30, 217],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [6, 30, 216],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 30, 215],
                    [6, 30, 217],
                    [6, 32, 217],
                    [0, 32, 223],
                    [0, 24, 219],
                    [5, 31, 214],
                    [6, 29, 217],
                    [6, 31, 217],
                    [6, 30, 215],
                    [6, 30, 216],
                    [5, 31, 215],
                    [6, 30, 216],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 30, 217],
                    [6, 30, 217],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 30, 217],
                    [6, 30, 217],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 30, 217],
                    [6, 30, 217],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 30, 217],
                    [6, 30, 217],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [6, 30, 217],
                    [6, 30, 217],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]]
                  )

wall = np.array([[[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [100, 90, 90],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 90, 90],
                  [0, 0, 0],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [103, 91, 91],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [101, 92, 92],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [101, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [103, 91, 91],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [101, 92, 92],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [101, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [101, 92, 92],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [101, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [100, 92, 92],
                  [101, 92, 92],
                  [101, 92, 92],
                  [101, 92, 92],
                  [101, 92, 92],
                  [100, 92, 92],
                  [101, 92, 92],
                  [101, 92, 92],
                  [102, 92, 92],
                  [101, 92, 92],
                  [101, 92, 92],
                  [101, 92, 92],
                  [100, 92, 92],
                  [101, 92, 92],
                  [101, 92, 92],
                  [101, 92, 92],
                  [101, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [102, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [101, 93, 93],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [101, 93, 93],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [101, 93, 93],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [100, 92, 92],
                  [104, 92, 92],
                  [103, 90, 90],
                  [103, 90, 90],
                  [103, 90, 90],
                  [103, 95, 95],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [103, 92, 92],
                  [103, 92, 92],
                  [101, 91, 91],
                  [102, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [101, 92, 92],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [101, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [101, 92, 92],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [101, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [101, 92, 92],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [101, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [100, 92, 92],
                  [104, 92, 92],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [100, 92, 92],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [98, 98, 98],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [100, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [106, 97, 97],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 92, 92],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [103, 91, 91],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [100, 94, 94],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [102, 92, 92],
                  [103, 91, 91],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [100, 90, 90],
                  [100, 92, 92],
                  [100, 92, 92],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [101, 93, 93],
                  [100, 92, 92],
                  [100, 92, 92],
                  [100, 90, 90],
                  [0, 0, 0],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]]])
