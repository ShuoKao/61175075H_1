# 61175075H_1
Computer Vision Assignment 1

1. Coins
	In this case, I built a system to detect gesture. If user open palm, the system will start to detect the coin, then draw the edge of coins. However, I used open-cv library HoughCircle to detect the circles. It somehow detect the wrong objects eg. finger, head on the coins. I limit the circle area if radius >= 20 the shape of circles will be painted.

2. Cards
	I used findcontour to detect the shape of cards. If it detect the rectangle, the edge of cards will be painted.
	The problem in this case is that findcontour can't detect the rectangle correctly every time.
	
3. Dice
	In this case, i didn't find a efficientive solution to detect each side of dice.
