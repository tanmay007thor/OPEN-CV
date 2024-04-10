import cv2

cap = cv2.VideoCapture(0)


num = 0

while cap.isOpened():
   
    succes, img = cap.read()
    print(img)
    k = cv2.waitKey(5)
    if num == 15 :
        break
    if k == 27:
        break
    
    cv2.imwrite('images/img' + str(num) + '.png', img)
    print("image saved!")
    num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()