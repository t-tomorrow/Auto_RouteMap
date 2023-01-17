import glob
import func, func2
import numpy as np

#入力する画像データ（元画像群）
input_files = glob.glob(".\\images100\\*")
#関数addの結果ファイルの出力先
output_dir = ".\\study/kmean3_num2"             #研究用
#ピクセルの黒の度合いを何分割(n分割)して、最頻値・期待値を求めるか（階級:0~n-1, n~2n-1, ... ~255)
range_n = 10
#ピクセルの黒の度合いの閾値
threshold_bottom = 130

model = "./study/model.jpg"

#レーダー画像から船舶を検出する
#func.findShip(input_files, output_dir, range_n, threshold_bottom)
#func.sub(input_files, model, output_dir, threshold_bottom=130, threshold_top=170, num=2, k=3)






#モジュール
#---------------------------------------------------------------------------------------------------------------
#入力する画像データ（元画像）
input_files = glob.glob(".\\images2_40piece\\*")
#入力するモデルデータ(".\\modelMode_max.jpg" or ".\\modelMode_expect.jpg")
model_file = ".\\images2_001\\model.jpg"
#関数subの結果ファイルの出力先
output_sub_destination = ".\\outputMax_sub_130-170_k3median5"
#output_sub_destination = glob.glob(".\\outputMax_sub_130-170_k5median5\\*")
#関数addの結果ファイルの出力先
output_add_destination = ".\\outputMax_add3_130-170_median"

#モデル画像作成
#func.average(files)
#func.sum(files)
#func.mode(input_files, 10)


#元画像からモデル画像を差分した画像を作成する
threshold_bottom = 130
threshold_top = 170

num = 5         #平滑化の回数
k_size = 3      #平滑化する際のフィルターのサイズ
#func.sub(input_files, model_file, output_sub_destination, threshold_bottom, threshold_top, num, k_size)

#元画像に差分した画像を加えた画像を作成する
#func.add(input_files, output_sub_destination, output_add_destination)



#一つの画像に対する画素値を求める（確認用）
#func.dot_image(".\\output5_k5median3.jpg", "output5_k5median3.txt", "output5_k5median3_2.jpg")


#平滑化処理
#func.median_filter(".\\output5_k5median3.jpg", "output5_k5median4.jpg")

#レーダー画像から船を四角で囲む
out_dir = ".\\plot\k3median1"
func2.exploler_dir(glob.glob(".\\outputMax_sub_130-170_k3median1\*"), out_dir)

#func2.exploler(".\\outputMax_sub_130-170_k5median3\\out_img_14.jpg", "outputMax_sub_130-170_k5median3\\out_img_15.jpg")
#func2. encircle(".\\output5_k5median3.jpg", ".\\a")

vecA = np.array([1, 2])
p1 = [2, 2] #Trueがかえってこればよい
p2 = [5, 1] #False
#func2.isInArea(vecA, p2)

"""
# 画像同士で合成したい場合・・・
a = glob.glob(".\\a\\*")
b = glob.glob(".\\b\\*")
func.add(a, b, ".\\c")
"""
#----------------------------------------------------------------------------------------------------
