import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

#avg_imgの初期化用
once = True

#ファイルの数をカウント
file_count = 0


def findShip(base_files, out_dir, range_n=10, threshold=0):
    global once, file_count
    files = base_files
    os.mkdir(out_dir)
    pattern_num = int(255/range_n)+1

    print("ピクセルごとの黒色度合い（最頻値）を計算中・・・")
    #フォルダ内にある画像を使う
    for file in tqdm(files):
        file_count += 1
        file = cv2.imread(file, 0)
        if once == True:
            #初期化
            mode_img_3d = np.array([[[0 for i in range(pattern_num)] for j in range(len(file[0]))] for k in range(len(file))])
            mode_img_2d = np.array([[0 for i in range(len(file[0]))] for j in range(len(file))])
            once = False

        #わかりやすいように黒地に白→白地に黒へ変換し、黒の濃さを調べる
        for i in range(len(file)):
            for j in range(len(file[0])):
                dot_level = int(255 - file[i][j])
                mode_img_3d[i][j][int(dot_level/range_n)] += 1

    print("\nモデル図作成中・・・")
    for i in range(len(file)):
        for j in range(len(file[0])):
            #expected_valie：期待値
            expected_value_sum = 0
            expected_value = 0
            min_value = 0
            max_value = 0
            for k in range(pattern_num):
                min_value = range_n * k
                max_value = (range_n * (k+1)-1)
                if max_value > 255:
                    max_value = 255

                expected_value_sum += int((min_value + max_value)/2) * mode_img_3d[i][j][k]
            expected_value = int(expected_value_sum / file_count)
            mode_img_2d[i][j] = 255 - expected_value

    cv2.imwrite(out_dir + "\model.jpg",mode_img_2d)

    #-------------------------------------------------------------------------------------------
    once = True
    file_count = 0

    if not ((0<=threshold) and (threshold<=255)):
        print("閾値の設定が不適切です。閾値を0に設定します。")
        threshold = 0

    base_model = mode_img_2d

    print("差分画像作成中・・・")
    for file in tqdm(files):
        file_count += 1
        file = cv2.imread(file, 0)
        if once == True:
            #初期化
            out_img = np.array([[0 for i in range(len(file[0]))] for j in range(len(file))])

            if threshold != 0:
                sub_dir = out_dir + "\\sub_files_over" + str(threshold)
            else:
                sub_dir = out_dir + "\\sub_files"
            os.mkdir(sub_dir)

            once = False

        #わかりやすいように黒地に白→白地に黒へ変換し、黒の濃さで操作を行う
        for i in range(len(file)):
            for j in range(len(file[0])):                    
                out_img[i][j] = int(255 - file[i][j]) - int(255 - base_model[i][j])
                    
                if out_img[i][j] < 0:
                    out_img[i][j] = 0

                #閾値設定時
                if out_img[i][j] >= int(threshold):
                    out_img[i][j] = 255 - out_img[i][j]
                else:
                    out_img[i][j] = 255

        
        cv2.imwrite(sub_dir + "\\out_img_" + "{:03}".format(file_count) + ".jpg", out_img)


    #-------------------------------------------------------------------------------------------
    # 加算画像（現在使用しないのでコメントアウト）
    """
    file_count = 0
    once = True
    base = base_files
    over = glob.glob(sub_dir + "\\*")

    if once == True:
        if threshold != 0:
            add_dir = out_dir + "\\add_files_over" + str(threshold) + "\\"
        else:
            add_dir = out_dir + "\\add_files\\"
        os.mkdir(add_dir)
        once = False

    print("元画像と差分画像との合成画像作成中・・・")
    for base_file, over_file in tqdm(zip(base, over), total=len(base)):
        file_count += 1
        base_file = cv2.imread(base_file)
        over_file = cv2.imread(over_file)
        base_file = cv2.cvtColor(base_file, cv2.COLOR_BGR2RGB)
        over_file = cv2.cvtColor(over_file, cv2.COLOR_BGR2RGB)

        #わかりやすいように黒地に白→白地に黒へ変換し、黒の濃さを足していく
        for i in range(len(base_file)):
            for j in range(len(base_file[0])):
                if 255 - int(over_file[i][j][0]) > 0:
                    base_file[i][j][0] = 255 - over_file[i][j][0]   #R
                    base_file[i][j][1] = 0                          #G
                    base_file[i][j][2] = 0                          #B

        base_file = cv2.cvtColor(base_file, cv2.COLOR_RGB2BGR)

        cv2.imwrite(add_dir + "add_img_" + "{:03}".format(file_count) + ".jpg", base_file)
    """
    print("\nEND")



    #調査画像からmode関数で作った画像の画素を引く
#threshold:閾値。デフォルトは０。
def sub(files, model, out, threshold_bottom=0, threshold_top=255, num=3, k=3):
    global once, file_count
    if not ((0<=threshold_bottom) and (threshold_bottom<=255) and (0<=threshold_top) and (threshold_top<=255)):
        print("閾値の設定が不適切です。閾値を0に設定します。")
        threshold_bottom = 0
        threshold_top = 255

    base_file = cv2.imread(model, 0)
    #フォルダ内にある画像を使う
    print("差分画像作成中・・・")
    os.mkdir(out)
    for file in tqdm(files):
        file_count += 1
        file = cv2.imread(file, 0)
        if once == True:
            #初期化
            out_img = np.array([[0 for i in range(len(file[0]))] for j in range(len(file))])
            once = False

        if file.shape == base_file.shape:
            #わかりやすいように黒地に白→白地に黒へ変換し、黒の濃さで操作を行う
            for i in range(len(file)):
                for j in range(len(file[0])):                    
                    out_img[i][j] = int(255 - file[i][j]) - int(255 - base_file[i][j])
                    #line = "i:" + str(i) + ",j:" + str(j) + "  :" + str(out_img[i][j])
                    #print(line)
                    
                    if out_img[i][j] < 0:
                        out_img[i][j] = 0

#---------------------------------------------------------------------------------
                    #閾値設定時
                    if  (int(threshold_bottom) <= out_img[i][j]) and (out_img[i][j] <= int(threshold_top)):
                        out_img[i][j] = 255 - out_img[i][j]
                    else:
                        out_img[i][j] = 255
                    #line = "i:" + str(i) + ",j:" + str(j) + "  :" + str(255 - out_img[i][j]) + "\n"
                    #print(line)
                    #open_file(line, "sub.txt")
#----------------------------------------------------------------------------------
        else:
            print("画像のサイズが異なっています。")


        #平滑化処理を(num)回行う
        for i in range(num):
            out_img = cv2.medianBlur(out_img.astype(np.float32), ksize=k)
            #out_img = cv2.GaussianBlur(out_img.astype(np.float32), ksize=(3,3), sigmaX=1.3)
        
        cv2.imwrite(out + "\\out_img_" + "{:02}".format(file_count) + ".jpg", out_img)
    print("\nEND")


#船と思われる部分に色を加える
def add(base, over, output):
    global file_count
    #フォルダ内にある画像を使う
    print("元画像と差分画像との合成画像作成中・・・")
    os.mkdir(output)
    for base_file, over_file in tqdm(zip(base, over), total=len(base)):
    #for base_file, over_file in zip(base, over):
        file_count += 1
        base_file = cv2.imread(base_file)
        over_file = cv2.imread(over_file)
        base_file = cv2.cvtColor(base_file, cv2.COLOR_BGR2RGB)
        over_file = cv2.cvtColor(over_file, cv2.COLOR_BGR2RGB)

        cv2.imshow('image',base_file)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('image',over_file)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if base_file.shape == over_file.shape:
            #わかりやすいように黒地に白→白地に黒へ変換し、黒の濃さを足していく
            for i in range(len(base_file)):
                for j in range(len(base_file[0])):
                    #print("1:   i:" + str(i) + ",j:" + str(j) + ":   " + str(255 - over_file[i][j]))
                    if 255 - int(over_file[i][j][0]) > 0:
                        base_file[i][j][0] = 255 - over_file[i][j][0]   #R
                        base_file[i][j][1] = 0                        #G
                        base_file[i][j][2] = 0                        #B
                    #print("2:   i:" + str(i) + ",j:" + str(j) + ":   " + str(255 - base_file[i][j]))
                    #print("\n")
        else:
            print("画像のサイズが異なっています。")

        base_file = cv2.cvtColor(base_file, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output + "\\last_img_" + "{:02}".format(file_count) + ".jpg", base_file)
    print("\nEND")










#最頻値/期待値を使ってモデル図を作成
def mode(files, n):  #n: 0~n
    global once, file_count
    pattern_num = int(255/n)+1
    #フォルダ内にある画像を使う
    for file in files:
        file_count += 1
        print("################################################" + str(file_count))
        file = cv2.imread(file, 0)
        if once == True:
            #初期化
            mode_img_3d = np.array([[[0 for i in range(pattern_num)] for j in range(len(file[0]))] for k in range(len(file))])
            mode_img_2d = np.array([[0 for i in range(len(file[0]))] for j in range(len(file))])
            once = False

        #わかりやすいように黒地に白→白地に黒へ変換し、黒の濃さを調べる
        for i in range(len(file)):
            for j in range(len(file[0])):
                #print(int(file[i][j]))
                dot_level = int(255 - file[i][j])

                #line = "i:" + str(i) + ",    j:" + str(j) + ":  " + str(dot_level) + "\n"
                #open_file(line, "img_dot_" + str(file_count) + ".txt")

                #print(dot_level)
                mode_img_3d[i][j][int(dot_level/n)] += 1
                #if file_count == 3:
                    #print(mode_img_3d[i][j])
                    #line2 = "i:" + str(i) + ",    j:" + str(j) + ":  " + str(mode_img_3d[i][j]) + "\n"
                    #open_file(line2, "mode_img_3d.txt")

    for i in range(len(file)):
        for j in range(len(file[0])):
            #maxで考えたとき
            """
            max = 0
            max_k = 0
            for k in range(pattern_num):
                if max <= mode_img_3d[i][j][k]:
                    max = mode_img_3d[i][j][k]
                    max_k = k
            
            #濃度の範囲のうち最小値と最大値を２で割り、範囲の中間値をとる
            middle_num = int(((max_k*n)+((max_k+1)*n-1))/2)
            if middle_num > 255:
                middle_num = 255
            mode_img_2d[i][j] = 255 - middle_num
            #print(mode_img_2d[i][j])
            """

            #expected_valie：期待値
            expected_value_sum = 0
            expected_value = 0
            min_value = 0
            max_value = 0
            for k in range(pattern_num):
                min_value = n * k
                max_value = (n * (k+1)-1)
                if max_value > 255:
                    max_value = 255

                expected_value_sum += int((min_value + max_value)/2) * mode_img_3d[i][j][k]
            expected_value = int(expected_value_sum / file_count)
            mode_img_2d[i][j] = 255 - expected_value

    cv2.imwrite("mode_img2.jpg",mode_img_2d)        #mode_img.jpgはmaxを使ったとき, mode_img2.jpgは期待値を使ったとき
    #cv2.imshow("", mode_img_2d.astype(np.float32))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print("\nEND")


#調査画像からmode関数で作った画像の画素を引く
#threshold:閾値。デフォルトは０。
def sub(files, model, out, threshold_bottom=0, threshold_top=255, num=3, k=3):
    global once, file_count
    if not ((0<=threshold_bottom) and (threshold_bottom<=255) and (0<=threshold_top) and (threshold_top<=255)):
        print("閾値の設定が不適切です。閾値を0に設定します。")
        threshold_bottom = 0
        threshold_top = 255

    base_file = cv2.imread(model, 0)
    #フォルダ内にある画像を使う
    print("差分画像作成中・・・")
    #os.mkdir(out)
    for file in tqdm(files):
        file_count += 1
        file = cv2.imread(file, 0)
        if once == True:
            #初期化
            out_img = np.array([[0 for i in range(len(file[0]))] for j in range(len(file))])
            once = False

        if file.shape == base_file.shape:
            #わかりやすいように黒地に白→白地に黒へ変換し、黒の濃さで操作を行う
            for i in range(len(file)):
                for j in range(len(file[0])):                    
                    out_img[i][j] = int(255 - file[i][j]) - int(255 - base_file[i][j])
                    #line = "i:" + str(i) + ",j:" + str(j) + "  :" + str(out_img[i][j])
                    #print(line)
                    
                    if out_img[i][j] < 0:
                        out_img[i][j] = 0

#---------------------------------------------------------------------------------
                    #閾値設定時
                    if  (int(threshold_bottom) <= out_img[i][j]) and (out_img[i][j] <= int(threshold_top)):
                        out_img[i][j] = 255 - out_img[i][j]
                    else:
                        out_img[i][j] = 255
                    #line = "i:" + str(i) + ",j:" + str(j) + "  :" + str(255 - out_img[i][j]) + "\n"
                    #print(line)
                    #open_file(line, "sub.txt")
#----------------------------------------------------------------------------------
        else:
            print("画像のサイズが異なっています。")


        #平滑化処理を(num)回行う
        for i in range(num):
            out_img = cv2.medianBlur(out_img.astype(np.float32), ksize=k)
            #out_img = cv2.GaussianBlur(out_img.astype(np.float32), ksize=(3,3), sigmaX=1.3)
        
        cv2.imwrite(out + "\\out_img_" + "{:02}".format(file_count) + ".jpg", out_img)
    print("\nEND")


#船と思われる部分に色を加える
def add(base, over, output):
    global file_count
    #フォルダ内にある画像を使う
    print("元画像と差分画像との合成画像作成中・・・")
    os.mkdir(output)
    for base_file, over_file in tqdm(zip(base, over), total=len(base)):
    #for base_file, over_file in zip(base, over):
        file_count += 1
        base_file = cv2.imread(base_file)
        over_file = cv2.imread(over_file)
        base_file = cv2.cvtColor(base_file, cv2.COLOR_BGR2RGB)
        over_file = cv2.cvtColor(over_file, cv2.COLOR_BGR2RGB)

        cv2.imshow('image',base_file)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('image',over_file)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if base_file.shape == over_file.shape:
            #わかりやすいように黒地に白→白地に黒へ変換し、黒の濃さを足していく
            for i in range(len(base_file)):
                for j in range(len(base_file[0])):
                    #print("1:   i:" + str(i) + ",j:" + str(j) + ":   " + str(255 - over_file[i][j]))
                    if 255 - int(over_file[i][j][0]) > 0:
                        base_file[i][j][0] = 255 - over_file[i][j][0]   #R
                        base_file[i][j][1] = 0                        #G
                        base_file[i][j][2] = 0                        #B
                    #print("2:   i:" + str(i) + ",j:" + str(j) + ":   " + str(255 - base_file[i][j]))
                    #print("\n")
        else:
            print("画像のサイズが異なっています。")

        base_file = cv2.cvtColor(base_file, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output + "\\last_img_" + "{:02}".format(file_count) + ".jpg", base_file)
    print("\nEND")



def dot_image(file, save_text, save_file):
    file = cv2.imread(file)
    file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
    for i in tqdm(range(len(file))):
        for j in range(len(file[0])):
            #黒の濃さの度合いを標準出力
            line = "i:" + str(i) + ",j:" + str(j) + "  :" + str(255-file[i][j]) + "\n"

            if  (i%50==0 or j%50==0) or (i%10==0 and j%10==0):
                file[i][j][0] = 0
                file[i][j][1] = 150
                file[i][j][2] = 0
            #print(line)
            open_file(line, save_text)
    file = cv2.cvtColor(file, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_file, file)


#平滑化処理（メディアンフィルタ）
def median_filter(file, save_file):
    file = cv2.imread(file, 0)
    img_med = cv2.medianBlur(file, ksize=5)
    #cv2.imshow('img', img_med)
    cv2.imwrite(save_file, img_med)


def open_file(line, file_name):
    with open(file_name, "a") as f:
        f.write(line)
