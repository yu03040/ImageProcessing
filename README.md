# ImageProcessing
画像処理に関して学んだことを実装していきます。

# プログラム
- Source
  - diff_filter.py（正方形のみの画像のエッジ検出）
	 グレースケール変換してからエッジ検出する。
	 
     |元画像|グレースケール変換した画像|
     |:---:|:---:|
     |![yasai256original](https://github.com/yu03040/ImageProcessing/assets/131416689/0f644056-8e87-4b03-83b7-9afcbc785be5)|![yasai256gray](https://github.com/yu03040/ImageProcessing/assets/131416689/1c723da3-a1e7-4181-b7b6-cca0ab7c5c84)|

	二つの手法
	
	 |roll 関数で周期シフトして垂直水平の両方向のエッジ検出|線形代数を使って垂直水平の両方向のエッジ検出|
     |:---:|:---:|
     |![yasai256gray_1vh](https://github.com/yu03040/ImageProcessing/assets/131416689/1a69aac4-d758-4f71-9755-dcf5e8378f1c)|![yasai256gray_2vh](https://github.com/yu03040/ImageProcessing/assets/131416689/6bb54e98-2137-4e1f-aadb-03d395539acb)|

	- 1.vh と 2.vh は違う手法だが、出力画像は同じになる。(誤差がゼロ)
	
  - diff_filter2.py（長方形を含む画像のエッジ検出）
    - diff_filter2 は diff_filter の改良版で正方形と長方形の画像に対応している。
    グレースケール変換してからエッジ検出する。
	 
	 |元画像|グレースケール変換した画像|
     |:---:|:---:|
     |![yasai512_256original](https://github.com/yu03040/ImageProcessing/assets/131416689/e4d99cbe-5b62-4753-9861-6796e36fbee8)|![yasai512_256gray](https://github.com/yu03040/ImageProcessing/assets/131416689/2af3973d-7612-4589-9199-ad24851b9709)|

	 二つの手法
	
	 |roll 関数で周期シフトして垂直水平の両方向のエッジ検出|線形代数を使って垂直水平の両方向のエッジ検出|
     |:---:|:---:|
     |![yasai512_256_1vh](https://github.com/yu03040/ImageProcessing/assets/131416689/913c6032-ba59-4e8d-9ca9-026127db5010)|![yasai512_256_2vh](https://github.com/yu03040/ImageProcessing/assets/131416689/95b57d70-c00d-4cd9-a10c-4828f6d9b588)|
	
	- 1.vh と 2.vh は違う手法だが、出力画像は同じになる。(誤差がゼロ)