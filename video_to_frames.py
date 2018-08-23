import cv2
import skvideo.io
import skvideo.datasets

def extractImages(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success, image = vidcap.read()
      if success:
          cv2.imwrite(pathOut + "/frame%d.jpg" % count, image)     # save frame as JPEG file
          count += 1
    if count > 0:
        #print('%d images were extracted successfully' % count)
        pass
    else:
        print('Images extraction failed.')
    return count


def extractImages2(pathIn, pathOut):

    videogen = skvideo.io.vreader(skvideo.datasets.bigbuckbunny())
    for frame in videogen:
        print(frame.shape)

# if __name__=="__main__":
#     print("aba")
#     extractImages("spity.mp4", "frames/")