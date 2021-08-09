def zoneCreator(x, y):
        """
        # Define Zones
        topright = [[0,0], [1280,720]]
        topleft = [[0,0], [-1280,720]]
        bottomleft = [[0,0], [-1280, -720]]
        bottomright = [[0,0], [1280, -720]]

        currpos_x = x
        currpos_y = y
        """

        # Define borderchecks
        if (x == 0 or y == 0):
            return "border"

        if (y > 0):
            if (x > 0):
                return "topright"
            else:
                return "topleft"

        if (y < 0):
            if (x > 0):
                return "bottomright"
            else:
                return "bottomleft"