import React, { useEffect } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Image } from 'react-native';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, { FadeInDown, ZoomIn } from 'react-native-reanimated';

export default function LoginScreen() {
    const router = useRouter();

    const handleLogin = () => {
        router.replace('/(tabs)');
    };

    return (
        <View style={styles.container}>
            <LinearGradient
                colors={['#000000', '#1C1C1E']}
                style={styles.background}
            />

            <View style={styles.content}>
                <View style={styles.logoContainer}>
                    <Animated.View entering={ZoomIn.delay(200).duration(600)} style={styles.logo}>
                        <Ionicons name="wallet" size={40} color={Colors.primary} />
                    </Animated.View>
                    <Animated.Text entering={FadeInDown.delay(400).duration(600)} style={styles.appName}>Arthya</Animated.Text>
                    <Animated.Text entering={FadeInDown.delay(500).duration(600)} style={styles.tagline}>Your Personal Finance Assistant</Animated.Text>
                </View>

                <View style={styles.authContainer}>
                    <Animated.View entering={FadeInDown.delay(700).duration(600)}>
                        <TouchableOpacity style={styles.biometricButton} onPress={handleLogin}>
                            <Ionicons name="finger-print" size={64} color={Colors.primary} />
                            <Text style={styles.biometricText}>Tap to Login</Text>
                        </TouchableOpacity>
                    </Animated.View>

                    <Animated.Text entering={FadeInDown.delay(800).duration(600)} style={styles.orText}>OR</Animated.Text>

                    <Animated.View entering={FadeInDown.delay(900).duration(600)} style={{ width: '100%' }}>
                        <TouchableOpacity style={styles.pinButton} onPress={handleLogin}>
                            <Text style={styles.pinButtonText}>Use PIN</Text>
                        </TouchableOpacity>
                    </Animated.View>
                </View>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.background,
    },
    background: {
        position: 'absolute',
        left: 0,
        right: 0,
        top: 0,
        bottom: 0,
    },
    content: {
        flex: 1,
        justifyContent: 'space-around',
        alignItems: 'center',
        padding: 40,
    },
    logoContainer: {
        alignItems: 'center',
    },
    logo: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: '#333',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 20,
        borderWidth: 2,
        borderColor: Colors.primary,
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
        elevation: 8,
    },
    appName: {
        fontSize: 32,
        fontWeight: 'bold',
        color: Colors.text,
        letterSpacing: 2,
    },
    tagline: {
        color: Colors.textDim,
        marginTop: 8,
        fontSize: 14,
    },
    authContainer: {
        alignItems: 'center',
        width: '100%',
    },
    biometricButton: {
        alignItems: 'center',
        marginBottom: 40,
    },
    biometricText: {
        color: Colors.primary,
        marginTop: 16,
        fontSize: 16,
        fontWeight: '500',
    },
    orText: {
        color: Colors.textDim,
        marginBottom: 40,
    },
    pinButton: {
        width: '100%',
        paddingVertical: 16,
        borderRadius: 30,
        backgroundColor: Colors.card,
        alignItems: 'center',
        borderWidth: 1,
        borderColor: Colors.border,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
        elevation: 3,
    },
    pinButtonText: {
        color: Colors.text,
        fontSize: 16,
        fontWeight: '600',
    },
});
