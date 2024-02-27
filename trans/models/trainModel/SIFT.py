
from pandas import ExcelWriter
from baseFewShotMatcher import BaseFewShotMatcher

# from google.colab.patches import cv2_imshow
class sift(BaseFewShotMatcher):

    def predict(self, tar, templates, tempbox,_):
        '''
            you should be return box, class, confidence score
        '''
        # convert images to grayscale
        # print(len(templates))
        predictions = []
        pred_in_shot = []
        for temp in templates:
            score = 1
            time_pred = 0
            t1 = time.time()
            # net = cv2.xfeatures2d.SIFT_create()
            net = cv2.SIFT_create()
            # print(temp.shape)
            # find the keypoints and descriptors with SIFT
            kp1, des1 = net.detectAndCompute(temp, None)
            kp2, des2 = net.detectAndCompute(tar, None)
            if des1 is None or des2 is None:
                predictions.append([-1, -1, -1, -1, -1, -1])
            else:
                # create feature matcher
                bf = cv2.BFMatcher(crossCheck=True)
                # match descriptors of both images
                matches = bf.match(des1, des2)
                # sort matches by distance
                # matches = bf.match(des1,des2)
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = matches[:20]
                score = [good_matches[0].distance for i in range(len(good_matches))]
                score = np.average(score) / 100

                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                if src_pts is None or dst_pts is None or len(src_pts) < 4 or len(dst_pts) < 4:
                    predictions.append([-1, -1, -1, -1, -1, -1])
                else:
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    matchesMask = mask.ravel().tolist()
                    h, w = temp.shape[:2]
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                    if pts is None or M is None:
                        predictions.append([-1, -1, -1, -1, -1, -1])
                    else:
                        dst = cv2.perspectiveTransform(pts, M)

                        # print("*************************************************")
                        x, y = int(np.int32(dst).tolist()[0][0][0]), int(np.int32(dst).tolist()[0][0][1])
                        x1, y1 = int(np.int32(dst).tolist()[2][0][0]), int(np.int32(dst).tolist()[2][0][1])
                        # target=tar.copy()
                        # cv2.rectangle(target,(x,y), (x1,y1), 255,1)
                        # cv2_imshow( target)
                        # cv2.waitKey()
                        # print("*************************************************")
                        # img3 = cv2.drawMatches(temp,kp1,tar,kp2,good_matches, None,**draw_params)
                        # cv2_imshow( img3)
                        # cv2.waitKey()
                        # x_sift,y_sift=(int(np.int32(dst).tolist()[0][0][0]),int(np.int32(dst).tolist()[0][0][1]))
                        # x2_sift,y2_sift=(int(np.int32(dst).tolist()[2][0][0]),int(np.int32(dst).tolist()[2][0][1]))
                        t2 = time.time()
                        predictions.append([x, y, x1, y1, -1 * score, (t2 - t1)])
                        # predictions.append(pred_in_shot)
        # self.visualize(tar, predictions)
        predictions.sort(reverse=True, key=itemgetter(4))
        # print(predictions)
        return predictions
