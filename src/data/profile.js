export const profile = {
  name: "Xinzhe Zhou",
  chineseName: "周昕哲",
  title: "Undergraduate Student in Automation",
  affiliation: "School of Automation and Intelligent Sensing, Shanghai Jiao Tong University",
  location: "Shanghai, China",
  email: "zhou_xinzhe@sjtu.edu.cn",
  personalEmail: "zhouxinzhe0414@gmail.com",
  github: "https://github.com/Zhouxinzhe",
  avatar: "./img/zhouxinzhe.jpg",
  heroImage: "./img/home-bg-geek.jpg",
  interests: [
    "Multi-agent systems",
    "Formation control",
    "Robotics",
    "Graph neural networks",
    "Geometric deep learning",
    "Generative Model",
    "Flow Matching",
    "Motion Planning",
    "Reinforcement Learning"
  ],
  metrics: [
    { label: "GPA", value: "4.12 / 4.3" },
    { label: "Core average", value: "94.678" },
    { label: "Major rank", value: "1 / 93" }
  ],
  honors: [
    "Principal Investigator, NSFC Young Student Basic Research Program (Undergraduate), 2025",
    "Outstanding Bachelor's Thesis, Shanghai Jiao Tong University Class of 2026 (top 1% university-wide)",
    "National Scholarship, 2024",
    "Pan Wenyuan Scholarship, 2023",
    "Shanghai First Prize, National Undergraduate Electronics Design Contest, 2024",
    "RoboCup China Robot Competition awards, 2024",
    "National VEX Robotics Competition, University First Prize, 2023",
    "Outstanding Student and Outstanding League Member, SJTU"
  ],
  publications: [
    {
      year: "2025",
      title: "Formation Maneuver Control Based on the Augmented Laplacian Method",
      authors: "Xinzhe Zhou, Xuyang Wang, Xiaoming Duan, Yuzhu Bai, Jianping He",
      venue: "IEEE Conference on Decision and Control (CDC), 2025",
      status: "CDC 2025",
      summary:
        "Proposes an augmented Laplacian framework for 2-D and 3-D formation maneuvers, enabling translation, scaling, and arbitrary-orientation rotation with reduced neighbor requirements.",
      links: [
        { label: "arXiv", url: "https://arxiv.org/abs/2505.05795" },
        { label: "DOI", url: "https://doi.org/10.48550/arXiv.2505.05795" }
      ]
    },
    {
      year: "2025",
      title:
        "A General and Efficient SE(3)-Equivariant Graph Framework: Encoding Symmetries with Complete Differential Invariants and Frames",
      authors: "Xuyang Wang, Xinzhe Zhou, Tao Xu, Xiaoming Duan, Jianping He",
      venue: "Manuscript / under review",
      status: "Manuscript",
      summary:
        "Develops a general SE(3)-equivariant graph framework that encodes symmetry through complete differential invariants and local frames.",
      links: [{ label: "Paper", url: "https://openreview.net/forum?id=JN2h0CpdCO" }]
    },
    {
      year: "2026",
      title: "SE(3)-Equivariant Flow Matching with Gaussian Process Priors for Geometric Trajectory Prediction",
      authors: "Xuyang Wang, Xinzhe Zhou, Xiaoming Duan, Jianping He",
      venue: "International Conference on Machine Learning (ICML), 2026",
      status: "ICML 2026",
      summary:
        "Combines SE(3)-equivariant flow matching with Gaussian process priors for geometry-aware trajectory prediction.",
      links: [{ label: "Paper", url: "https://openreview.net/forum?id=FR0ZFbkGnX" }]
    },
    {
      year: "2026",
      title: "BridgeFlow: Fast and Robust SE(2)-Equivariant Motion Planning with Flow Matching",
      authors: "Xinzhe Zhou et al.",
      venue: "IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2026",
      status: "IROS 2026",
      summary:
        "Studies fast and robust SE(2)-equivariant motion planning through flow matching. arXiv version not yet uploaded.",
      links: []
    }
  ],
  experience: [
    {
      time: "2025",
      title: "Formation maneuver control",
      description:
        "First-author work connecting multi-agent control, 3-D rotation, and dynamic agent reconfiguration through augmented Laplacian methods."
    },
    {
      time: "2025",
      title: "SE(3)-equivariant graph neural networks",
      description:
        "Research around complete differential frames, invariants, and geometry-aware graph representation learning."
    },
    {
      time: "2023 - 2024",
      title: "Undergraduate innovation project",
      description:
        "Trajectory-based imitation learning algorithm verification platform."
    }
  ]
};
