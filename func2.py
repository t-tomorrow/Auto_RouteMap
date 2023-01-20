import os
import cv2
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from tqdm import tqdm
import glob
import random

cnt = 0
plot_count = 0
first_flag = True
different_color = 0
 

# 進行方向として正しいように変更するための変数
static_point_num = []                     #ルートごとのポイントにおける先頭の値を保存
route_cnt = 0
Route_num = [[] for i in range(1000)]            #ルートの数がどのくらい出現するのかわからないため、1000個の配列を用意
route_lastPoint = 0  #個別のルートを表示する際に使用。最終地点を保存する変数

#１枚目のベクトルデータを保存
save_vec = [[0 for i in range(2)] for j in range(50)]

# 選択した個別の船舶の航路を表示するための変数
draw_point = []
draw_Arrow_point = []


#複数の画像を読み込んで探索し、探索結果をベースファイルに書き込む
#入力ファイルは平滑化後のファイル群
def exploler_dir(files, out_dir):
    base_files = glob.glob(".\\images\*")

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    it1 = iter(files)
    it2 = iter(base_files)
    file1 = next(it1)
    b_file1 = next(it2)

    for i in tqdm(range(len(files) -1)):
        file2 = next(it1)
        b_file2 = next(it2)
        route_map = exploler(file1, file2, b_file2, out_dir)
        file1 = file2

    draw_ArrowImg(route_map, out_dir)



    




#２枚の画像から船の動向を推測し、船を追う
def exploler(file1, file2, b_file=".\\images2_001\model.jpg", out_dir = ".\\plot"):
    save_dir = out_dir
    base_file = b_file

    #os.mkdir(save_dir)
    contours1 = encircle(file1, save_dir)
    contours2 = encircle(file2, save_dir)

    # 総合評価結果用
    #eval_matrix: simとdistの値が小さいほど精度が高いので、eval_matrixの値が小さいほどよいとする。
    eval_matrix = np.empty((len(contours1), len(contours2)))

    sim_matrix = similarity(contours1, contours2, save_dir)         #船同士の類似度を返す
    dist_matrix = distance(contours1, contours2, save_dir)          #船同士の距離を返す
    arg_matrix = angle(file1, contours1, contours2, save_dir)       #船が+-30°以内に存在するか（１か０）を返す

    #print("----------------------------------------------")

    for i, (sim, dist, arg) in enumerate(zip(sim_matrix, dist_matrix, arg_matrix)):
        for j in range(len(sim)):
            if arg[j] == 0:
                eval_matrix[i][j] = 0
            else:
                if abs(dist[j] -50) < 50 and (sim[j] < 1.0):
                    eval_matrix[i][j] = sim[j] * abs(dist[j] -50) * 100        #simもdistも小さい値のほうがいいのでかけてみた。距離：平均50くらい？
                else:
                    eval_matrix[i][j] = 0
    #print(eval_matrix)


    #print("----------------------------------------------")
    # sort[len(contours1)][len(contours2)][2]: 精度が高い順に並び替える
    # 第一要素：船の番号。第二要素：二つ目の画像に映る船。第三要素：可能性の高い船の番号とその評価度。
    sort = np.array([[[0 for i in range(2)] for j in range(len(eval_matrix[0]))] for k in range(len(eval_matrix))])
    sorted1 = np.array([[[0 for i in range(2)] for j in range(len(eval_matrix[0]))] for k in range(len(eval_matrix))])
    sorted2 = np.array([[[0 for i in range(2)] for j in range(len(eval_matrix[0]))] for k in range(len(eval_matrix))])

    #精度が高い順に並べる（先頭側に評価0があるので、それはsorted2で並べなおす）
    for i in range(len(eval_matrix)):
        for j, eval in enumerate(eval_matrix[i]):
            sort[i][j][0] = j
            sort[i][j][1] = eval
        sorted1[i] = sorted(sort[i], key=lambda x: x[1])
        cnt = 0
        for j in range(len(eval_matrix[i])):
            if sorted1[i][j][1] != 0:
                sorted2[i][cnt] = sorted1[i][j]
                cnt += 1

    #print("----------------------------------------------")
    #print(sorted2)

    #予想した船を保存する
    predict_ship = []
    predicted_ship1 = []
    predicted_ship2 = []

    cnt = 0
    #print("---------------予想結果--------------------")
    for i in range(len(eval_matrix)):
        #print(str(i) + "の船")
        j = 0
        while((j != len(sorted2[i])) and (sorted2[i][j][1] != 0)):
            #print("    " + str(i) + " -----> " + str(sorted2[i][j][0]) + "  \t予想度合：" + str(sorted2[i][j][1]))
            predict_ship.append([i, sorted2[i][j][0], sorted2[i][j][1]])

            j += 1
            cnt += 1
        #print("\n")

    #print("--------------------------------------------")
    #船を予測度合い順に並び替え
    #現在の船と次に進んだ船が決定すれば、その船番号を保存
    num_nowShip = []
    num_nextShip = []
    cnt = 0
    for i in range(len(predict_ship)):
        predicted_ship1 = sorted(predict_ship, key=lambda x: x[2])
        #print(str(predicted_ship1[i][0]) + " -----> " + str(predicted_ship1[i][1]) + "  \t予想度合：" + str(predicted_ship1[i][2]))
        
        #予想度合いが3000以下の場合に設定
        if ((predicted_ship1[i][0] not in num_nowShip) and (predicted_ship1[i][1] not in num_nextShip) and (predicted_ship1[i][2] < 3000)):
            num_nowShip.append(predicted_ship1[i][0])
            num_nextShip.append(predicted_ship1[i][1])
            predicted_ship2.append([predicted_ship1[i][0], predicted_ship1[i][1], predicted_ship1[i][2]])
            print(str(predicted_ship2[cnt][0]) + " -----> " + str(predicted_ship2[cnt][1]) + "  \t予想度合：" + str(predicted_ship2[cnt][2]))
            cnt += 1


    global first_flag
    global save_vec
    #船の進行方向として正しい動作をさせるための関数
    save_route(predicted_ship2, contours1, contours2)

    save_vec.clear()
    save_vec = [[[] for i in range(2)] for j in range(len(contours1))]
    for i in range(len(predicted_ship2)):
        save_vec[i][0] = [predicted_ship2[i][0], predicted_ship2[i][1]]
        x1, y1, w1, h1 = cv2.boundingRect(contours1[predicted_ship2[i][0]])
        x2, y2, w2, h2 = cv2.boundingRect(contours2[predicted_ship2[i][1]])
        save_vec[i][1] = np.array([int((x2 + w2/2) - (x1 + w1/2)), int((y2 + h2/2) - (y1 + h1/2))])

    #予想された船同士を線でつなぎ、モデル図に反映させる
    #line(base_file, file2, predicted_ship2, contours1, contours2, save_dir)
    
    #応急処置
    if first_flag == True:
        route_map_Img = Route_map(base_file, file2, predicted_ship2, contours1, contours2, save_dir)        # <---panasionic用
    else:
        route_map_Img = Route_map(out_dir + "\\result.jpg", file2, predicted_ship2, contours1, contours2, save_dir)
        #Route_map(out_dir + "\\result.jpg", file2, predicted_ship2, contours1, contours2, save_dir)        # <---panasionic用
    first_flag = False

    return route_map_Img

            

# ルート保存関数
# route_directions : 取得した進路の始点と終点を格納した配列変数
def save_route(route_directions, contours1, contours2):
    global Route_num
    global route_cnt
    global static_point_num
    route_cnt_preSum = route_cnt   #以前までのルート数（固定）
    route_cnt_Sum = route_cnt + len(route_directions)   #現在の最終ルート数


    if first_flag == True:
        static_point_num.append(0)  #最初の値は0
        for i in range(len(route_directions)):
            Route_num[i].append(route_directions[i][0])
            Route_num[i].append(len(contours1) + route_directions[i][1])
            route_cnt += 1
        static_point_num.append(len(contours1))
            
    
    else:
        static_point_num.append(static_point_num[len(static_point_num) -1] + len(contours1))   #カウント保存用
        for i in range(len(route_directions)):
            exist_preShip_Flag = False      #フレーム前の画像に同一船が存在するか？

            for j in range(route_cnt_preSum):
                # フレーム前の船の終点とフレーム後の船の始点が同じならば、紐づける
                if (Route_num[j][len(Route_num[j])-1] - static_point_num[len(static_point_num)-2]) == route_directions[i][0]:
                    #print("A==" + str(len(Route_num[j])-1))
                    Route_num[j].append(static_point_num[len(static_point_num) -1] + route_directions[i][1])

                    exist_preShip_Flag = True

                #異なれば、新しくデータを残す
            if exist_preShip_Flag == False:
                route_cnt += 1
                Route_num[route_cnt -1].append(static_point_num[len(static_point_num) -2] + route_directions[i][0])
                Route_num[route_cnt -1].append(static_point_num[len(static_point_num) -1] + route_directions[i][1])
                
    print("\n=================航路データ===========================")
    print("先頭データ：" + str(static_point_num))
    for i in range(route_cnt):
        print(str(i) + "：" + str(Route_num[i]))









#船を囲むための関数
def encircle(file, save_dir):
    global plot_count
    plot_count += 1

    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 輪郭の検出
    contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 輪郭を１つずつ書き込んで出力
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), cv2.LINE_4)
        cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 1, cv2.LINE_AA)

    #cv2.imwrite(save_dir + "\\result" + str(plot_count) + ".jpg", img)
    
    #draw_contours(im, contours)
    return contours


# 類似度を計算する。
def similarity(contours1, contours2, save_dir):
    #fig = plt.figure()

    matrix = np.empty((len(contours1), len(contours2)))
    for i, j in np.ndindex(*matrix.shape):
        matrix[i, j] = cv2.matchShapes(contours1[i], contours2[j], cv2.CONTOURS_MATCH_I1, 0)

        #類似度の値としてふさわしくない（発散している値）のとき
        #類似度の値に発散した値が入るのは縦線一本の線などの場合
        if "e" in str(matrix[i, j]):
            #print("i=" + str(i) + "\t,j=" + str(j) + "\t:船の形が線で表示されているため、適切な類似度ではありません")
            matrix[i, j] = 10000

    """
    # 行列を可視化する。
    fig, ax = plt.subplots(figsize=(14, 14))
    ax = sns.heatmap(
        matrix, annot=True, cmap="Reds", ax=ax, fmt=".2f", annot_kws={"size": 15}
    )
    """

    #fig.savefig(save_dir + "\\similarity.jpg")
    #plt.show()
    return matrix


# 距離を計算する。
def distance(contours1, contours2, save_dir):
    #fig = plt.figure()

    matrix = np.empty((len(contours1), len(contours2)))
    for i, j in np.ndindex(*matrix.shape):
        x1, y1, w1, h1 = cv2.boundingRect(contours1[i])
        x2, y2, w2, h2 = cv2.boundingRect(contours2[j])
        point1 = ((x1 + w1/2), (y1 + h1/2))
        point2 = ((x2 + w2/2), (y2 + h2/2))
        matrix[i, j] = math.sqrt(math.fsum( (x - y)*(x - y) for x, y in zip(point1, point2)))

    """
    # 行列を可視化する。
    fig, ax = plt.subplots(figsize=(14, 14))
    ax = sns.heatmap(
        matrix, annot=True, cmap="Blues", ax=ax, fmt=".2f", annot_kws={"size": 10}
    )
    """

    #fig.savefig(save_dir + "\\distance.jpg")
    #plt.show()
    return matrix


# 角度を計算する。
def angle(file1, contours1, contours2, save_dir):
    img = cv2.imread(file1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    matrix = np.empty((len(contours1), len(contours2)))


    # 映っている船それぞれの傾きを調べる
    for i in range(len(contours1)):
        cnt1 = 0
        cnt2 = 0
        threshold = 100
        isNotOverThreshold = True
        notGradientFlag = False
        now_x, now_y, now_w, now_h = cv2.boundingRect(contours1[i])

        while isNotOverThreshold:
            for j in range(int(now_h/2)):
                for k in range(int(now_w/2)):
                    #船の傾きがどちら向きに傾いているかを計算する
                    #船を四角で囲み、それを４等分に分けたうち、上部の２つでどちらに多く画素が含まれているか
                    if np.all(255 - img[now_y +j][now_x +k] >= threshold):
                        cnt1 += 1
                    if np.all(255 - img[now_y +j][(now_x + now_w -1) -k] >= threshold):
                        cnt2 += 1

            if cnt1 == 0 and cnt2 == 0:
                threshold -= 10

                #縦線一本が画像に残っている場合は、ここを通る
                #閾値を下げていってもcnt1とcnt2は常に0となるため、ループから抜けるためフラグをfalseにする
                if threshold == 0:
                    #print(str(i) + "番目の角度を正しく測ることができませんでした。")
                    isNotOverThreshold = False      
            else:
                isNotOverThreshold = False

        # 左右の画素値にあまり差が見られなかったら・・・
        if (cnt1>=3 and cnt2>=3) and abs(cnt1-cnt2) < 3:
            notGradientFlag = True

        #船の傾きをベクトルで指定
        #if (cnt1 != 0 and cnt2 == 0) or (cnt1/cnt2) >= 1.1:
        if (cnt2 != 0) and (cnt1 > cnt2) and (cnt1/cnt2) >= 1.1:
            #print("i:" + str(i) + ",    左向きに傾いている")
            if notGradientFlag == True:
                if now_h < now_w:
                    vec = np.array([-now_w/2, -now_h/4])
                else:
                    vec = np.array([-now_w/4, -now_h/2])
            else:
                vec = np.array([-now_w/2, -now_h/2])
        #elif (cnt1 == 0 and cnt2 != 0) or (cnt1/cnt2) <= 0.9:
        elif (cnt1 < cnt2) and (cnt1/cnt2) <= 0.9:
            #print("i:" + str(i) + ",    右向きに傾いている")
            if notGradientFlag == True:
                if now_h < now_w:
                    vec = np.array([now_w/2, -now_h/4])
                else:
                    vec = np.array([now_w/4, -now_h/2])
            else:
                vec = np.array([now_w/2, -now_h/2])
        else:
            #print("i:" + str(i) + ",    あまり傾きがない")
            if now_h < now_w:
                vec = np.array([1, 0])
            elif now_w < now_h:
                vec = np.array([0, 1])
            else:
                vec = np.array([0, 0])

        #print(vec)        

        # 船の傾きが表せなかった場合・・・
        if np.all(vec == 0):
            #print(str(i) + "番目の船は回転して探索します。")
            #rotate_exploler
            #matrix[i, point] = False
            matrix[i] = np.array([False for point in range(len(contours2))])
        else:
            for point in range(len(contours2)):
                now_position = [now_x + now_w/2, now_y + now_h/2]
                next_x, next_y, next_w, next_h = cv2.boundingRect(contours2[point])
                next_position = [next_x + next_w/2, next_y + next_h/2]

                pass_Flag = False
                if first_flag == False:
                    for l in range(len(save_vec)):
                        if len(save_vec[l][0]) != 0 and i == save_vec[l][0][1]:
                            if np.dot(save_vec[l][1], vec) < 0:
                                matrix[i, point] = False
                                pass_Flag = True
                    if pass_Flag == False:
                        matrix[i, point] = isInArea(img, save_dir, now_position, vec, next_position)

        #arg = math.degrees(math.atan(h/w))
        #print("i:" + str(i) + ",    arg:" + str(arg))
    
    """
    fig = plt.figure()

    # 行列を可視化する。
    fig, ax = plt.subplots(figsize=(14, 14))
    ax = sns.heatmap(
        matrix, annot=True, cmap="Reds", ax=ax, fmt=".2f", annot_kws={"size": 15}
    )
    """

    #fig.savefig(save_dir + "\\isInArea.jpg")
    #plt.show()
    return matrix

        




# ある点が、ある方向ベクトルに対する+-30°の領域内に存在するかの確認
# img:                  角度が書かれた画像を出力する用
# now_pos, next_pos:    座標値
def isInArea(img, save_dir, now_pos, vecA, next_pos):
    theta = 30
    arg = [theta, -theta]
    norm_of_A = np.linalg.norm(vecA)
    unit_vecA = vecA / norm_of_A
    R_vec = np.empty((len(arg), 2))

    for i in range(len(arg)):
        t = np.deg2rad(arg[i])

        # 回転行列
        R = np.array([[np.cos(t), -np.sin(t)],
                      [np.sin(t),  np.cos(t)]])

        R_vec[i] = np.dot(R, unit_vecA)

    # 連立方程式を使って領域内に存在するかを確認
    # s(x1, y1) + t(x2, y2) = (x, y): OA + OB = OPのとき、s>=0, t>=0で領域内に存在
    A = np.array([[R_vec[0][0], R_vec[1][0]],
                  [R_vec[0][1], R_vec[1][1]]])
    
    #点Pを原点0に移動して、ベクトル計算を行う
    P = np.array([next_pos[0] - now_pos[0], next_pos[1] - now_pos[1]])
    
    s, t = np.linalg.solve(A, P)


    cv2.line(img, (int(now_pos[0] - 100*R_vec[0][0]), int(now_pos[1] - 100*R_vec[0][1])), (int(now_pos[0] + 100*R_vec[0][0]), int(now_pos[1] + 100*R_vec[0][1])), (255, 0, 0), 1)
    cv2.line(img, (int(now_pos[0] - 100*R_vec[1][0]), int(now_pos[1] - 100*R_vec[1][1])), (int(now_pos[0] + 100*R_vec[1][0]), int(now_pos[1] + 100*R_vec[1][1])), (255, 0, 0), 1)
    #cv2.imwrite(save_dir + "\\result3.jpg", img)




    if (s>=0 and t>=0) or (s<=0 and t<= 0): #点Pが領域内に存在する条件
        #print("     点Pは領域内に存在します。")
        return True
    else:
        #print("     点Pは領域内に存在しません。")
        return False


#船同士で矢印線をつなぐ（２つ）
def line(base_file, next_file, predict_result, contours1, contours2, save_dir):
    base_img = cv2.imread(base_file)
    #base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

    t = os.path.getmtime(next_file)
    update_time = datetime.datetime.fromtimestamp(t)
    date, time = str(update_time).split()

    for i in range(len(predict_result)):
        x1, y1, w1, h1 = cv2.boundingRect(contours1[predict_result[i][0]])
        x2, y2, w2, h2 = cv2.boundingRect(contours2[predict_result[i][1]])
        cv2.arrowedLine(base_img, (int(x1+w1/2), int(y1+h1/2)), (int(x2+w2/2), int(y2+h2/2)), (0, 255, 0), 3)

    cv2.putText(base_img, date.ljust(20), (int(len(base_img[0])*5/7), int(len(base_img) *19/20 -20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
    cv2.putText(base_img, time.ljust(20), (int(len(base_img[0])*5/7), int(len(base_img) *19/20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
    #base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

    global cnt
    cnt += 1
    cv2.imwrite(save_dir + "\\result.jpg", base_img)
    #cv2.imwrite(".\\abc.jpg", base_img)


practice_cnt = 0
#船同士で線をつなぐ（全体）
def Route_map(base_file, next_file, predict_result, contours1, contours2, save_dir):
    base_img = cv2.imread(base_file)
    #base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

    #t = os.path.getmtime(next_file)
    #update_time = datetime.datetime.fromtimestamp(t)
    #date, time = str(update_time).split()

    draw_route_Flag = False
    draw_route = 170
    global route_lastPoint
    global draw_point
    global draw_Arrow_point

    if len(Route_num[draw_route]) != 0:
        draw_route_Flag = True

        if Route_num[draw_route][len(Route_num[draw_route])-1] == route_lastPoint:
            draw_route_Flag = False

        route_lastPoint = Route_num[draw_route][len(Route_num[draw_route]) -1]

    for i in range(len(predict_result)):
        if (draw_route_Flag == True) and (predict_result[i][0] == Route_num[draw_route][len(Route_num[draw_route]) -2] - static_point_num[len(static_point_num) -2]):
            x1, y1, w1, h1 = cv2.boundingRect(contours1[predict_result[i][0]])
            x2, y2, w2, h2 = cv2.boundingRect(contours2[predict_result[i][1]])
            #cv2.arrowedLine(base_img, (int(x1+w1/2), int(y1+h1/2)), (int(x2+w2/2), int(y2+h2/2)), (0, 0, 255), 3, tipLength=0.3)
            draw_point = [int(x1+w1/2), int(y1+h1/2), int(x2+w2/2), int(y2+h2/2)]
            draw_Arrow_point.append(draw_point)

        else:
            x1, y1, w1, h1 = cv2.boundingRect(contours1[predict_result[i][0]])
            x2, y2, w2, h2 = cv2.boundingRect(contours2[predict_result[i][1]])
            cv2.line(base_img, (int(x1+w1/2), int(y1+h1/2)), (int(x2+w2/2), int(y2+h2/2)), (0, 153, 119) , 3)   

    """
    if (draw_route_Flag == True) and (route_lastPoint == 47 or route_lastPoint == 57):
        global practice_cnt
        cv2.imwrite(save_dir + "\result.jpg", base_img)
        cv2.imwrite(save_dir + "\practice_" + str(practice_cnt) + ".jpg", base_img)
        practice_cnt += 1
    """
    cv2.imwrite(save_dir + "/result.jpg", base_img)
    return base_img




def draw_ArrowImg(route_map, save_dir):
    arrow_img = np.array([[[255 for i in range(3)] for j in range(len(route_map[0]))] for k in range(len(route_map))], dtype = 'uint8')
    #result_img = np.array([[[0 for i in range(3)] for j in range(len(route_map[0]))] for k in range(len(route_map))])
    
    # 選択した航路を後から上書きして表示させる
    for i in range(len(draw_Arrow_point)):
        cv2.arrowedLine(arrow_img, (draw_Arrow_point[i][0], draw_Arrow_point[i][1]), (draw_Arrow_point[i][2], draw_Arrow_point[i][3]), (25, 165, 0), 3, tipLength=0.3)

    cv2.imwrite(save_dir + "/arrow.jpg", arrow_img)
    result_img = cv2.addWeighted(route_map, 0.3, arrow_img, 0.7, 2.2)
    cv2.imwrite(save_dir + "/result.jpg", result_img)





# 2D座標設定関数
def coordinate(axes, range_x, range_y, grid = True,
               xyline = True, xlabel = "x", ylabel = "y"):
    axes.set_xlabel(xlabel, fontsize = 16)
    axes.set_ylabel(ylabel, fontsize = 16)
    axes.set_xlim(range_x[0], range_x[1])
    axes.set_ylim(range_y[0], range_y[1])
    if grid == True:
        axes.grid()
    if xyline == True:
        axes.axhline(0, color = "gray")
        axes.axvline(0, color = "gray")

# 2Dベクトル描画関数
def visual_vector(axes, loc, vector, color = "red"):
    axes.quiver(loc[0], loc[1],
              vector[0], vector[1], color = color,
              angles = 'xy', scale_units = 'xy', scale = 1)
