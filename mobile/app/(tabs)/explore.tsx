import React, { useState } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, ScrollView, Alert, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { Colors } from '@/constants/Colors';
import { useAuth } from '@/contexts/AuthContext';
import Animated, { FadeInDown } from 'react-native-reanimated';

export default function ProfileScreen() {
  const { user, logout } = useAuth();
  const router = useRouter();
  const [loggingOut, setLoggingOut] = useState(false);

  const handleLogout = () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Logout',
          style: 'destructive',
          onPress: async () => {
            setLoggingOut(true);
            try {
              await logout();
              router.replace('/login');
            } catch (error) {
              console.error('Logout error:', error);
              Alert.alert('Error', 'Failed to logout. Please try again.');
            } finally {
              setLoggingOut(false);
            }
          },
        },
      ],
      { cancelable: true }
    );
  };

  const getInitials = (name?: string) => {
    if (!name) return 'U';
    return name.split(' ').map(n => n[0]).join('').toUpperCase().substring(0, 2);
  };

  const menuItems = [
    { icon: 'person-outline', label: 'Edit Profile', onPress: () => {} },
    { icon: 'notifications-outline', label: 'Notifications', onPress: () => {} },
    { icon: 'shield-checkmark-outline', label: 'Privacy & Security', onPress: () => {} },
    { icon: 'card-outline', label: 'Payment Methods', onPress: () => {} },
    { icon: 'help-circle-outline', label: 'Help & Support', onPress: () => {} },
    { icon: 'information-circle-outline', label: 'About', onPress: () => {} },
  ];

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        {/* Header */}
        <Animated.View entering={FadeInDown.delay(100).duration(500)} style={styles.header}>
          <Text style={styles.headerTitle}>Profile</Text>
        </Animated.View>

        {/* Profile Card */}
        <Animated.View entering={FadeInDown.delay(200).duration(500)} style={styles.profileCard}>
          <View style={styles.avatarContainer}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>{getInitials(user?.name)}</Text>
            </View>
            <TouchableOpacity style={styles.editAvatarButton}>
              <Ionicons name="camera" size={16} color="#fff" />
            </TouchableOpacity>
          </View>
          <Text style={styles.userName}>{user?.name || 'User'}</Text>
          <Text style={styles.userEmail}>{user?.email || 'No email'}</Text>
          {user?.occupation && (
            <View style={styles.occupationBadge}>
              <Ionicons name="briefcase-outline" size={14} color={Colors.primary} />
              <Text style={styles.occupationText}>
                {user.occupation.charAt(0).toUpperCase() + user.occupation.slice(1).replace('_', ' ')}
              </Text>
            </View>
          )}
        </Animated.View>

        {/* Menu Items */}
        <Animated.View entering={FadeInDown.delay(300).duration(500)} style={styles.menuSection}>
          {menuItems.map((item, index) => (
            <TouchableOpacity
              key={index}
              style={styles.menuItem}
              onPress={item.onPress}
              activeOpacity={0.7}
            >
              <View style={styles.menuItemLeft}>
                <View style={styles.menuIconContainer}>
                  <Ionicons name={item.icon as any} size={22} color={Colors.primary} />
                </View>
                <Text style={styles.menuItemText}>{item.label}</Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color={Colors.textDim} />
            </TouchableOpacity>
          ))}
        </Animated.View>

        {/* Logout Button */}
        <Animated.View entering={FadeInDown.delay(400).duration(500)} style={styles.logoutSection}>
          <TouchableOpacity
            style={styles.logoutButton}
            onPress={handleLogout}
            disabled={loggingOut}
            activeOpacity={0.7}
          >
            {loggingOut ? (
              <ActivityIndicator size="small" color="#FF453A" />
            ) : (
              <>
                <Ionicons name="log-out-outline" size={22} color="#FF453A" />
                <Text style={styles.logoutText}>Logout</Text>
              </>
            )}
          </TouchableOpacity>
        </Animated.View>

        {/* App Version */}
        <Animated.View entering={FadeInDown.delay(500).duration(500)} style={styles.versionSection}>
          <Text style={styles.versionText}>Arthya v1.0.0</Text>
          <Text style={styles.copyrightText}>© 2025 Arthya Financial Coach</Text>
        </Animated.View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    paddingTop: 60,
    paddingBottom: 100,
  },
  header: {
    paddingHorizontal: 24,
    marginBottom: 24,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: Colors.text,
  },
  profileCard: {
    alignItems: 'center',
    paddingVertical: 24,
    marginHorizontal: 24,
    marginBottom: 24,
    backgroundColor: Colors.card,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  avatarContainer: {
    position: 'relative',
    marginBottom: 16,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: Colors.background,
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  avatarText: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#000',
  },
  editAvatarButton: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: Colors.accent,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: Colors.card,
  },
  userName: {
    fontSize: 22,
    fontWeight: 'bold',
    color: Colors.text,
    marginBottom: 4,
  },
  userEmail: {
    fontSize: 14,
    color: Colors.textDim,
    marginBottom: 12,
  },
  occupationBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: `${Colors.primary}22`,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    gap: 6,
  },
  occupationText: {
    fontSize: 13,
    color: Colors.primary,
    fontWeight: '500',
  },
  menuSection: {
    marginHorizontal: 24,
    marginBottom: 24,
    backgroundColor: Colors.card,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: Colors.border,
    overflow: 'hidden',
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 16,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  menuItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  menuIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: `${Colors.primary}15`,
    justifyContent: 'center',
    alignItems: 'center',
  },
  menuItemText: {
    fontSize: 16,
    color: Colors.text,
    fontWeight: '500',
  },
  logoutSection: {
    marginHorizontal: 24,
    marginBottom: 24,
  },
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 16,
    backgroundColor: '#FF453A15',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#FF453A30',
  },
  logoutText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FF453A',
  },
  versionSection: {
    alignItems: 'center',
    paddingTop: 16,
  },
  versionText: {
    fontSize: 14,
    color: Colors.textDim,
    marginBottom: 4,
  },
  copyrightText: {
    fontSize: 12,
    color: Colors.textDim,
    opacity: 0.6,
  },
});
