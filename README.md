# SkyWarden ✈️

**AI-powered aerial image analysis that detects, identifies, and flags military aircraft in real time.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-8A2BE2)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B)
![PyTorch](https://img.shields.io/badge/Backend-PyTorch-EE4C2C)
![Status](https://img.shields.io/badge/Status-Hackathon%20Prototype-yellow)

SkyWarden is a computer-vision system that detects military aircraft in aerial images, classifies each as friendly or enemy, and raises an alert the moment a hostile aircraft is spotted.

## Table of Contents

- [Objective](#objective)
- [Architecture](#architecture)
- [Features](#features)
- [Demo](#demo)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Results](#results)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [Support](#support)
- [Contributors](#contributors)

## Objective

SkyWarden demonstrates how modern computer vision can automatically detect and classify military aircraft from aerial imagery while providing an immediate friend-or-enemy assessment for situational awareness. It combines a YOLOv8 detector, a Segment-Anything-assisted labeling pipeline, and a Streamlit front end into a single end-to-end demo — from raw aerial imagery to an annotated, alert-triggering result.

## Architecture

```
Input Image
     │
     ▼
YOLOv8 Detection + Classification
     │
     ▼
Friend / Enemy Mapping
     │
     ▼
Annotated Output + Alert
```

Detection and classification happen in a single YOLOv8 pass (one model predicts both the box and the aircraft type); the friend/enemy step is a separate rule-based lookup against the static allegiance lists in `app.py`. See [How It Works](#how-it-works) for the full pipeline, including preprocessing and training.

## Features

- **Aircraft detection** — locates aircraft in an aerial image and draws bounding boxes with class name and confidence score.
- **77-class recognition** — recognizes 77 distinct aircraft types, from fighters and bombers to transports and UAVs.
- **Friend/enemy classification** — cross-references each detected type against a curated allegiance list and colors boxes accordingly (green = friend, red = enemy, white = unrecognized).
- **Real-time alerts** — surfaces a clear on-screen warning the moment any enemy aircraft is detected.
- **Web interface** — drag-and-drop image upload via Streamlit, no setup beyond installing dependencies.
- **Reproducible training pipeline** — scripts to go from raw labeled data to a trained YOLOv8 checkpoint, including SAM-assisted auto-labeling.


