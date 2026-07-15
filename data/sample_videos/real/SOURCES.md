# Real Surveillance Video Sources

All sources below are free, no-login-required, and legal for educational/portfolio use.
Place downloaded files here and name them as suggested.

---

## Tier 1 — Stock Video Platforms (No Login, Instant MP4 Download)

These platforms allow direct browser download in MP4 without an account.

### Mixkit — Best for no-login, no-watermark CCTV clips
| Scene         | Search URL                                                         | Suggested Filename      |
|---------------|--------------------------------------------------------------------|-------------------------|
| CCTV general  | https://mixkit.co/free-stock-video/cctv/                          | cctv_mixkit.mp4         |
| Security cam  | https://mixkit.co/free-stock-video/security-camera/               | security_cam.mp4        |
| Pedestrian    | https://mixkit.co/free-stock-video/pedestrian/                    | pedestrian_mixkit.mp4   |

**License:** Mixkit Free License — commercial + non-commercial allowed, no attribution needed.

---

### Pixabay — Royalty-free, no attribution, no account needed for small sizes
| Scene                    | Search URL                                                                  | Suggested Filename       |
|--------------------------|-----------------------------------------------------------------------------|--------------------------|
| CCTV cameras             | https://pixabay.com/videos/search/cctv%20cameras/                         | cctv_pixabay.mp4         |
| Pedestrians              | https://pixabay.com/videos/search/pedestrians/                             | pedestrians_pixabay.mp4  |
| Corridor                 | https://pixabay.com/videos/search/corridor/                                | corridor_pixabay.mp4     |
| People walking in office | https://pixabay.com/videos/search/people%20walk%20in%20the%20office/      | office_pixabay.mp4       |
| Shopping mall            | https://pixabay.com/videos/search/walking%20around%20the%20mall/          | mall_pixabay.mp4         |
| CCTV for offices         | https://pixabay.com/videos/search/cctv%20for%20offices/                   | cctv_office_pixabay.mp4  |

**Specific clip:** People walking in a shopping mall (by Engin Akyurt):
https://www.pexels.com/video/people-walking-around-a-shopping-mall-20597684/

**License:** Pixabay Content License — free for commercial/non-commercial, no attribution required.

---

### Pexels — HD/4K, no account for standard quality
| Scene          | Search URL                                                        | Suggested Filename      |
|----------------|-------------------------------------------------------------------|-------------------------|
| CCTV           | https://www.pexels.com/search/videos/cctv/                       | cctv_pexels.mp4         |
| Shopping mall  | https://www.pexels.com/search/videos/shopping%20mall/            | mall_pexels.mp4         |
| Warehouse      | https://www.pexels.com/search/videos/warehouse/                  | warehouse_pexels.mp4    |
| People walking | https://www.pexels.com/search/videos/people%20walking/           | people_walking_pexels.mp4 |
| Pedestrians    | https://www.pexels.com/search/videos/pedestrians/                | pedestrian_pexels.mp4   |

**License:** Pexels License — free, no attribution required, commercial use allowed.

---

### Videezy — Creative Commons licensed clips
| Scene       | Search URL                                              | Suggested Filename     |
|-------------|---------------------------------------------------------|------------------------|
| Pedestrian  | https://www.videezy.com/free-video/pedestrian          | pedestrian_videezy.mp4 |
| Corridor    | https://www.videezy.com/free-video/corridor            | corridor_videezy.mp4   |
| CCTV 4K     | https://www.videezy.com/free-video/cctv-4k             | cctv_4k_videezy.mp4    |
| Pedestrians | https://www.videezy.com/free-video/pedestrians         | pedestrians_videezy.mp4|

**Note:** Some Videezy clips require free account for HD; SD is available without login.
**License:** Creative Commons + Open Source. Attribution required for CC clips.

---

## Tier 2 — Academic Surveillance Datasets (Real CCTV, Research-Grade)

### EPFL CVLAB Multi-Camera Pedestrian Dataset
- **URL:** https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/
- **Content:** Indoor lab + outdoor campus sequences, 4 synchronized camera views, real pedestrians
- **No login required.** Direct .tar.gz download from page.
- **Suggested filenames:** `epfl_indoor_cam1.mp4`, `epfl_campus_cam1.mp4`
- **License:** Free for research and education.

### EPFL WILDTRACK (7-Camera HD Dataset)
- **URL:** https://www.epfl.ch/labs/cvlab/data/data-wildtrack/
- **Content:** Outdoor plaza, 7 HD cameras, dense pedestrian crowds
- **License:** Free for academic use.

### MOT Challenge — MOT17 Benchmark
- **URL:** https://motchallenge.net/data/MOT17/
- **Content:** 14 sequences (static + moving cameras), shopping mall, street, corridor, parking
- **Registration:** Free account required.
- **Suggested filenames:** `mot17_mall.mp4`, `mot17_corridor.mp4`
- **License:** Free for research use.

### VIRAT Video Dataset
- **URL:** https://viratdata.org/
- **Content:** Outdoor surveillance, parking lots, building entrances, ground + aerial views
- **Registration:** Free account required via request form.
- **License:** Free for research and education.

---

## Tier 3 — Internet Archive (Public Domain)

### Surveillance Camera Man (Full, 720p)
- **URL:** https://archive.org/details/surveillance-camera-man-1-8-720p-hls.mp-4
- **Content:** Real-world candid surveillance-style footage, indoor/outdoor
- **Direct download:** Available as MP4 from archive.org download panel
- **License:** Public domain / free use.

---

## Download Tips

1. **Mixkit** is the easiest — click a video, hit "Free Download", choose resolution, done.
2. **Pixabay** — click video, scroll to "Free Download", pick `1280x720` or `1920x1080`, no account for lower resolutions.
3. **Pexels** — click video > "Free Download" > choose SD/HD (no account for SD).
4. **EPFL CVLAB** — click dataset name on page → direct `.tar.gz` with `.avi` sequences → convert with `ffmpeg -i input.avi output.mp4`.
5. **archive.org** — use the "Download" panel on the right side of any item page.

## Recommended "Best First Downloads" (Fastest to Acquire)

| Priority | Source  | Scene          | Why                                      |
|----------|---------|----------------|------------------------------------------|
| 1        | Mixkit  | CCTV / security cam | No login, no watermark, instant MP4  |
| 2        | Pixabay | Corridor / pedestrian | Royalty-free, no attribution needed |
| 3        | Pexels  | Mall / warehouse | High quality, free license             |
| 4        | EPFL    | Indoor pedestrian | Real CCTV, research-grade ground truth |
| 5        | MOT17   | Mall / street  | Gold-standard tracking benchmark data    |
