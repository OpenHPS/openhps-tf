#!/bin/bash
cd /opt/app
npm install --unsafe-perm
npm run test -- --grep "data.openhps.accuracy2"
