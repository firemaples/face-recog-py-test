import sys
import numpy as np
import time
from os import walk
import face_recognition

tolerance=0.6

path_recog = sys.argv[1]

print("\n")
print("% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %")
print("% % % Start to recognize [{}] with the tolerance: {} % % %".format(path_recog, tolerance))
print("% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %")
print("\n")

start_time=time.time()

knownFaceSetPath="FaceSet/"
fileNames = []
for (dirpath, dirnames, filenames) in walk(knownFaceSetPath):
    fileNames.extend(filenames)
    break

print("\n".join(map(str, fileNames)))
print("**** Finding {} file name(s) in '{}' spent {} secs".format(len(fileNames), knownFaceSetPath,time.time() - start_time) + " ****\n")

# Load the jpg files into numpy arrays
start_time=time.time()
knownImages = []
for fileName in fileNames:
    #print("Loading image: " + fileName)
    knownImages.append(face_recognition.load_image_file(knownFaceSetPath + fileName))
print("**** Loading {} image(s) of known faces spent {} secs ****\n".format(len(knownImages), (time.time() - start_time))) 

#biden_image = face_recognition.load_image_file("biden.jpg")
#obama_image = face_recognition.load_image_file("obama.jpg")
#unknown_image = face_recognition.load_image_file("obama2.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.

start_time=time.time()
face_encodings = []
for image in knownImages:
    face_encodings.append(face_recognition.face_encodings(image)[0])
print("**** Encoding {} image(s) of known faces spent {} secs ****\n".format(len(face_encodings), (time.time() - start_time)))

start_time=time.time()
image_recog = face_recognition.load_image_file(path_recog)
encoding_recogs = []
encoding_recogs.extend(face_recognition.face_encodings(image_recog))
print("**** Loading & encoding {} face(s) [{}] to recognize spent {} secs ****\n".format(len(encoding_recogs), path_recog, time.time() - start_time))

start_time=time.time()
face_encodings_np = np.array(face_encodings)
print("Start to computing the distance between {} known encoding(s) and {} unknown face(s)".format(len(face_encodings_np), len(encoding_recogs)))
for f in range(len(encoding_recogs)):
    nearest_name = ""
    nearest_distance = 999
    face_distances = face_recognition.face_distance(face_encodings_np, encoding_recogs[f])
    for i in range(len(face_distances)):
        d=face_distances[i]
        name=fileNames[i]
        print("Face[{}] distance {}: {}".format(f, name, d))
        if d < nearest_distance:
            nearest_name = name
            nearest_distance = d
    if nearest_distance <= tolerance:
        print("#####################################################################")
        print("#### Face[{}] MAY BE [{}]({}) ####".format(f, nearest_name, nearest_distance))
        print("#####################################################################")
    else:
        print("#####################################################################")
        print("#### Face[{}] is UNRECONIZED, the nearest one is [{}]({}) ####".format(f, nearest_name, nearest_distance))
        print("#####################################################################")
print("**** Computing distance(s) spent {} secs ****\n".format(time.time() - start_time))

#try:
#    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
#    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
#    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
#except IndexError:
#    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
#    quit()

#known_faces = [
#    biden_face_encoding,
#    obama_face_encoding
#]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
#results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

#print("Is the unknown face a picture of Biden? {}".format(results[0]))
#print("Is the unknown face a picture of Obama? {}".format(results[1]))
#print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))
