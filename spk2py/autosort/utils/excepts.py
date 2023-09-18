#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class GmmFitException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Error has occured durring GMM fitting"