#!/bin/bash
sudo -g docker /usr/bin/docker-wrapper build --network=host . --rm --pull --no-cache -t bert
