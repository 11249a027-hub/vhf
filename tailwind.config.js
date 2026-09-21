/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        vhf: {
          blue: '#2563EB',
          'blue-light': '#3B82F6',
          'blue-dark': '#1D4ED8',
          cyan: '#06B6D4',
          'cyan-light': '#22D3EE',
          pink: '#EC4899',
          'pink-light': '#F472B6',
          magenta: '#D946EF',
          indigo: '#6366F1',
          navy: '#0F172A',
          'navy-dark': '#020617',
          surface: '#F8FAFC',
          card: '#FFFFFF',
          border: '#E2E8F0',
          muted: '#64748B'
        }
      },
      boxShadow: {
        'glow-blue': '0 0 20px -3px rgba(37, 99, 235, 0.25)',
        'glow-pink': '0 0 20px -3px rgba(236, 72, 153, 0.25)',
        'glow-cyan': '0 0 20px -3px rgba(6, 182, 212, 0.25)',
        'card-elevated': '0 4px 20px -2px rgba(15, 23, 42, 0.06), 0 2px 6px -1px rgba(15, 23, 42, 0.04)'
      },
      backgroundImage: {
        'gradient-ai': 'linear-gradient(135deg, #2563EB 0%, #06B6D4 50%, #EC4899 100%)',
        'gradient-ai-subtle': 'linear-gradient(135deg, rgba(37, 99, 235, 0.06) 0%, rgba(6, 182, 212, 0.06) 50%, rgba(236, 72, 153, 0.06) 100%)',
        'gradient-card': 'linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%)'
      }
    },
  },
  plugins: [],
}
