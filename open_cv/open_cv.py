import cv2 as cv

path = "C:\\Users\\Raushan\\Downloads\\Code\\open_cv\\"

img = cv.imread(path + 'test_image.jpg')
cv.imshow('window', img)
cv.waitKey(0)
cv.destroyAllWindows()


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('window', gray)
cv.waitKey(0)
cv.destroyAllWindows()

# to save image
cv.imwrite('nature.png', gray)



cap = cv.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()










